# Franka Research 3 model

The generated `robots/fr3_franka_hand.urdf` and the arm collision meshes in
`meshes/collision/` come from Franka Robotics'
[`franka_description`](https://github.com/frankarobotics/franka_description)
release 2.7.1 (commit `2fd6aaa8c5e3bbfe857b8fb19978694cdc41e6ea`).

The URDF was generated for robot type `fr3` with the standard `franka_hand`.
It retains the official FR3 kinematics, inertials, dynamics, joint limits,
collision geometry, hand geometry, and `fr3_hand_tcp` frame. To avoid
duplicating roughly 26 MB of rendering-only data, its visual elements reuse
the repository's existing Franka visual meshes; collision checking uses the
vendored FR3 meshes.

The upstream Apache-2.0 `LICENSE` and `NOTICE` are included in this directory.
