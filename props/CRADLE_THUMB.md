# Pack Tool cradle thumb for the SO-101 fixed jaw

`pack_tool_cradle_thumb.stl` (mm, print as is). A clip-on cap for the last 12.6 mm of the gripper's **fixed** finger with a
transverse saddle on the pad side, so Pack Tool No. 01 rests in one place during the carry and the moving jaw can
close onto it to trap it. Bare plastic already works in tests; this makes the rest position repeatable so every
demo and every rollout puts the tool in the same spot.

| feature | value | why |
| --- | --- | --- |
| cap depth | 12.6 mm over the fingertip (model z 93.0 to 105.6) + 2 mm end cap | covers the tapered tip where the pad is |
| cavity | lofted from the measured finger sections: 10.0 x 12.8 mm at the base to 6.1 x 6.8 mm at the tip, +0.3 mm clearance per side | SO-ARM100 `wrist_roll_follower_so101_v1.stl` |
| walls | 2.0 mm; saddle wall 5.5 mm on the pad side (+x in the model) | 3.1 mm of material under the groove |
| saddle groove | cylinder R 2.9 mm, axis 0.5 mm above the face: width 5.7 mm, depth 2.4 mm, across the full finger thickness | tool is 0.206 in / 5.2 mm at its balance point; 85 % of the diameter sits below the lips |
| groove position | 6.1 mm from the fingertip, along the finger | tip clear for the moving jaw's pad to land beside/over the tool |
| retention | zip-tie slot 3.0 x 1.6 mm through both side walls at the open end | the taper makes the cap tighten as it is pushed on; the tie keeps it from walking off |
| volume | 2.1 cm3 | ~3 g in PETG |

Use: the tool's balance point is 2.160 in (54.9 mm) from the butt, where its thumb trough already is. Lay the tool
across the groove with the trough on the saddle; the trough's spherical seat (SR.125) and the groove together stop
it rolling and sliding. Close the moving jaw to within a few percent for the carry; open at the tray.

Print: PETG or TPU 95A (TPU grips the steel better), 0.2 mm layers, 100 % infill (it is tiny), groove facing up,
no supports needed if printed open-end down. Fit test first: push it on by hand; if it stops short, scale the model
2 % in x and y; if it is loose, print at 100 % and add the zip tie. The real finger carries a rubber pad the public
model does not include, which is what the 0.3 mm clearance is for.

Model source: trimesh + manifold booleans from the measured finger profile (see git history for the script in the
commit message); `pack_tool_cradle_thumb_views.png` are the three orthographic wireframes. A Fusion version can be
built from the same numbers when Fusion is open.
