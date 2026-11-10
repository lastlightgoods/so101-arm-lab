"""infra/: the compose stack, its .env contract, the image, and the Terraform mirror of it. No Docker, no AWS, no network."""
import os, re, shutil, subprocess
import pytest
import infrahelpers as H


def test_yaml_subset_reader_agrees_with_pyyaml():
    yaml = pytest.importorskip("yaml")
    text = H.read(H.COMPOSE)
    assert H.load_yaml_subset(text) == yaml.safe_load(text)


def test_services_ports_restart_and_health(services):
    assert set(services) == {"mlflow", "postgres", "s3", "s3-init"}
    for name in ("mlflow", "postgres", "s3"):
        assert services[name]["restart"] == "unless-stopped", name
        assert services[name]["healthcheck"]["test"][0] in ("CMD", "CMD-SHELL"), name
    assert services["s3-init"]["restart"] == "no"   # why: a one-shot init container must not be restarted forever
    assert services["mlflow"]["ports"] == ["${MLFLOW_PORT}:5050"]   # the host port is the one tools/mlflow_ui.sh and the field guide name
    assert services["s3"]["ports"] == ["${S3_PORT}:9000", "${S3_CONSOLE_PORT}:9001"]
    assert "ports" not in services["postgres"]   # why: nothing outside the stack needs Postgres; migrate goes through the MLflow API
    assert services["mlflow"]["depends_on"] == {"postgres": {"condition": "service_healthy"}, "s3-init": {"condition": "service_completed_successfully"}}
    assert services["s3-init"]["depends_on"] == {"s3": {"condition": "service_healthy"}}
    assert services["mlflow"]["healthcheck"]["test"] == ["CMD", "python", "/app/healthcheck.py"]


def test_volumes_are_named_and_declared(compose, services):
    declared = set(compose["volumes"])
    used = {v.split(":")[0] for s in services.values() for v in s.get("volumes", [])}
    assert used == declared == {"pgdata", "s3data", "s3logs"}
    assert "pgdata:/var/lib/postgresql/data" in services["postgres"]["volumes"] and "s3data:/data" in services["s3"]["volumes"]


def test_env_example_matches_every_compose_variable():
    referenced = set(H.COMPOSE_VAR.findall(H.read(H.COMPOSE)))
    keys = H.env_keys()
    assert referenced == set(keys), f"compose refs {sorted(referenced - set(keys))} missing from .env.example; dead keys {sorted(set(keys) - referenced)}"
    assert "$${POSTGRES_USER}" in H.read(H.COMPOSE)   # the escaped form must stay escaped (pg_isready reads the container's env)


def test_env_example_holds_placeholders_only():
    for k, v in H.env_keys().items():
        if any(w in k for w in ("PASSWORD", "SECRET")): assert v.startswith("change-me"), k
    assert H.env_keys()["MLFLOW_PORT"] == "5050" and H.env_keys()["MLFLOW_VERSION"] == re.search(r"^ARG MLFLOW_VERSION=(\S+)", H.read(H.DOCKERFILE), re.M).group(1)
    assert not os.path.exists(os.path.join(H.INFRA, ".env")) or "infra/.env" in H.read(os.path.join(H.ROOT, ".gitignore"))


def test_env_wiring_between_services(services):
    m, p, s, i = (services[k]["environment"] for k in ("mlflow", "postgres", "s3", "s3-init"))
    for k in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"): assert m[k] == p[k] == "${" + k + "}"
    assert m["POSTGRES_HOST"] == "postgres" and m["POSTGRES_PORT"] == "5432" and m["MLFLOW_PORT"] == "5050"
    assert m["MLFLOW_S3_ENDPOINT_URL"] == i["MLFLOW_S3_ENDPOINT_URL"] == "http://s3:9000"
    assert m["MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE"] == "path"   # why: boto3 would otherwise try bucket.s3:9000 virtual-host addressing
    assert m["AWS_ACCESS_KEY_ID"] == i["AWS_ACCESS_KEY_ID"] == s["RUSTFS_ACCESS_KEY"] == "${S3_ACCESS_KEY}"
    assert m["AWS_SECRET_ACCESS_KEY"] == i["AWS_SECRET_ACCESS_KEY"] == s["RUSTFS_SECRET_KEY"] == "${S3_SECRET_KEY}"
    assert m["MLFLOW_ARTIFACT_BUCKET"] == i["MLFLOW_ARTIFACT_BUCKET"] == "${MLFLOW_ARTIFACT_BUCKET}"
    assert services["mlflow"]["image"] == services["s3-init"]["image"] and services["mlflow"]["build"] == services["s3-init"]["build"]
    assert services["s3-init"]["command"] == ["python", "/app/s3_init.py"]


def test_image_and_entrypoint():
    df, ep = H.read(H.DOCKERFILE), H.read(H.ENTRYPOINT)
    assert re.search(r'pip install .*"mlflow==\$\{MLFLOW_VERSION\}".*psycopg2-binary==.*boto3==', df)
    for f in re.search(r"^COPY (.+) /app/$", df, re.M).group(1).split(): assert os.path.exists(os.path.join(H.INFRA, f)), f
    assert 'CMD ["/app/entrypoint.sh"]' in df and "USER mlflow" in df
    for flag in ('--backend-store-uri "postgresql+psycopg2://', '--artifacts-destination "s3://${MLFLOW_ARTIFACT_BUCKET}/"', "--serve-artifacts", "--host 0.0.0.0", '--port "${MLFLOW_PORT}"'):
        assert flag in ep, flag
    for path in (H.ENTRYPOINT, H.MIGRATE, H.MLFLOW_UI):
        r = H.sh_syntax_ok(path); assert r.returncode == 0, r.stderr
    assert os.access(H.ENTRYPOINT, os.X_OK) and os.access(H.MIGRATE, os.X_OK) and os.access(H.MLFLOW_UI, os.X_OK)


def test_mlflow_ui_sh_drives_both_servers():
    t = H.read(H.MLFLOW_UI)
    for sub in ("start)", "stop)", "status)", "up)", "down)", "logs)"): assert sub in t, sub
    assert "docker compose" in t and "docker-compose.yml" in t and "--wait" in t
    assert "down -v" not in t.split("down)")[1].split(";;")[0].split("echo")[0]   # why: plain `down` keeps the data; wiping is a deliberate extra


def test_terraform_variables_mirror_compose_env():
    tfvars = H.tf_variables()
    assert {k.lower() for k in H.SHARED} <= tfvars, sorted({k.lower() for k in H.SHARED} - tfvars)
    assert {"allowed_cidr", "aws_region", "desired_count", "cpu_architecture", "db_instance_class"} <= tfvars
    v = H.read(os.path.join(H.TF, "variables.tf"))
    assert re.search(r'variable "mlflow_version".*?default\s*=\s*"' + re.escape(H.env_keys()["MLFLOW_VERSION"]) + '"', v, re.S)
    assert re.search(r'variable "mlflow_port".*?default\s*=\s*5050', v, re.S)
    assert re.search(r'variable "postgres_password".*?sensitive\s*=\s*true', v, re.S)
    assert not re.search(r'variable "postgres_password".*?default', v.split('variable "mlflow_artifact_bucket"')[0].split('variable "postgres_password"')[1], re.S)


def test_terraform_task_env_is_what_the_entrypoint_reads():
    main = H.read(os.path.join(H.TF, "main.tf")); ep = H.read(H.ENTRYPOINT)
    task_env = set(re.findall(r'name\s*=\s*"([A-Z][A-Z0-9_]*)"', main))
    read_by_entrypoint = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:}]", ep)) - {"PW"}
    assert read_by_entrypoint - {"MLFLOW_WORKERS"} <= task_env, sorted(read_by_entrypoint - task_env)
    assert "POSTGRES_PASSWORD" in main.split("secrets = [")[1].split("]")[0] and 'valueFrom = aws_ssm_parameter' in main   # never a plain env value
    assert "MLFLOW_S3_ENDPOINT_URL" not in H.tf_code(main) and "AWS_ACCESS_KEY_ID" not in H.tf_code(main)   # real S3 via the task role


def test_terraform_shape_and_guard_rails():
    main = H.read(os.path.join(H.TF, "main.tf"))
    assert re.search(r"retention_in_days\s*=\s*14", main) and re.search(r"deletion_protection\s*=\s*false", main)
    assert re.search(r"multi_az\s*=\s*false", main) and re.search(r"allocated_storage\s*=\s*20", main) and re.search(r"publicly_accessible\s*=\s*false", main)
    assert re.search(r"cpu\s*=\s*256", main) and re.search(r"memory\s*=\s*512", main) and 'requires_compatibilities = ["FARGATE"]' in main
    for ing in H.tf_blocks("ingress", main): assert "0.0.0.0/0" not in ing and ("var.allowed_cidr" in ing or "security_groups" in ing)
    assert re.search(r"status\s*=\s*\"Enabled\"", main) and "noncurrent_version_expiration" in main and "block_public_acls" in main
    assert re.search(r"USD\s*[\d.]+/hour.*\d{1,2} Nov 2026", main, re.S)   # the one recurring cost is priced and dated in a comment
    assert '"s3:GetObject"' in main and "${aws_s3_bucket.artifacts.arn}/*" in main and '"s3:*"' not in main
    assert {"backend.tf.example", "main.tf", "outputs.tf", "variables.tf", "versions.tf"} <= set(os.listdir(H.TF))
    assert not os.path.exists(os.path.join(H.TF, "backend.tf")) or "backend.tf" in H.read(os.path.join(H.ROOT, ".gitignore"))


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_terraform_fmt_clean():
    r = subprocess.run(["terraform", "fmt", "-check", "-recursive", "-no-color"], cwd=H.TF, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_secrets_lan_ips_or_home_paths_in_infra():
    files = [os.path.join(d, f) for d, _, fs in os.walk(H.INFRA) for f in fs if ".terraform" not in d and f != ".env" and not f.endswith(".lock.hcl")]
    files += [H.MLFLOW_UI, os.path.join(H.ROOT, "docs", "mlops", "infra.md"), os.path.join(H.ROOT, "docs", "mlops", "colima.md")]
    bad = re.compile(r"/Users/[a-z]|/home/[a-z]|192\.168\.|10\.0\.\d+\.\d+|@gmail|AKIA[0-9A-Z]{16}")
    for f in files:
        m = bad.search(H.read(f)); assert not m, f"{f}: {m.group(0)}"


# ---------------------------------------------------------------- found on the first Colima start (10 Nov 2026, docs/mlops/colima.md)
def test_entrypoint_names_the_postgres_driver_the_image_installs():
    """SQLAlchemy 2.1 resolves a bare postgresql:// to psycopg 3; the image ships psycopg2-binary, so the URI must say +psycopg2.
    The first container crash-looped with "No module named 'psycopg'" until it did."""
    ep, df = H.read(H.ENTRYPOINT), H.read(H.DOCKERFILE)
    assert "psycopg2-binary==" in df and "psycopg[" not in df
    assert 'postgresql+psycopg2://' in ep and '"postgresql://' not in ep


def test_clients_of_the_proxied_s3_store_turn_presigned_transfers_off():
    """mlflow >= 3.17 clients ask the server whether its artifact store does presigned URLs and, when it does (S3), move every artifact
    through them; the local stack signs them for http://s3:9000, which only the containers resolve. Every lab client must say no."""
    want = ("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD")
    mig = H.read(H.MIGRATE)
    for k in want: assert f"{k}=false" in mig.split("\n", 1)[1], k
    assert "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=1" in mig   # the migration log is the lab's record; keep mlflow's "View run" lines out of it
    for env_sh in (os.path.join(H.ROOT, "env.sh"), os.path.join(H.ROOT, "jetson", "env.sh")):
        t = H.tf_code(H.read(env_sh))   # comments stripped: the export must be live, not explained
        for k in want: assert re.search(rf"^export .*{k}=\$\{{{k}:-false\}}", t, re.M), f"{env_sh}: {k}"
    for py in (os.path.join(H.ROOT, "jetson", "labtrack.py"), os.path.join(H.ROOT, "tools", "so101_policy.py")):
        t = H.read(py)
        for k in want: assert f'"{k}", "false"' in t, f"{py}: {k}"
        assert '"MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1"' in t, py


def test_colima_runbook_exists_and_is_linked():
    doc = os.path.join(H.ROOT, "docs", "mlops", "colima.md"); t = H.read(doc)
    for s in ("brew install colima docker docker-compose", "colima start", "cliPluginsExtraDirs", "brew services start colima", "down -v", "colima delete"): assert s in t, s
    assert "colima.md" in H.read(os.path.join(H.ROOT, "docs", "mlops", "infra.md"))
    assert "colima" in H.read(H.MLFLOW_UI).split("stack()")[1].split("}")[0]   # the no-docker hint points at the runbook's install line
