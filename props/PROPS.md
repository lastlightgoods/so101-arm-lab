# Standard props for the SO-101 lab

No official LeRobot kit exists; this is the set the public SO-101 datasets, the SmolVLA paper tasks, our simulator and
the course projects converge on. Everything is cheap and high-contrast for 640x480 webcams.

| Prop | Spec | Used by | Notes |
| --- | --- | --- | --- |
| Colored wooden cubes | 1 in / 25 mm hardwood, 6 colors (e.g. Learning Resources LER0136, 102 pc, ~$27) | our simulator (25 mm red cube), SmolVLA pick-place, course projects | the single most useful prop; red/blue/yellow/green give sorting tasks for free |
| Target disc | blue, Ø100 mm | our simulator (r = 5 cm) | print `props_sheet.pdf` at 100%, or a blue felt coaster; tape it to the mat |
| Transparent box | clear acrylic, ~10 x 10 x 6 cm | LeRobot tutorial ("pink lego brick into the transparent box"), SmolVLA SO-101 dataset | any clear food container works; the policy must see through it |
| LEGO Duplo bricks | 2x2 or 2x4, pink/bright | LeRobot SO-101 tutorial dataset | Duplo is easier to grasp than standard Lego for this gripper |
| Small bowl | ~12 cm, light color | cube-to-bowl community datasets | a cheap plastic cereal bowl |
| Bins | 3 small open bins, different colors | sorting tasks (course activity) | or three sheets of colored card as drop zones |
| Our lighter | BIC-style | the handoff task | keep for the real task; everything above is for the benchmark tasks |

Rules that matter more than the objects: fixed cameras, the cutting mat as the coordinate frame, randomize object
positions over a marked zone (10 episodes per position band), 50+ demos per task before training.
