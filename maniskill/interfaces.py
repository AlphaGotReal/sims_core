import torch

# Keys follow the pattern "{name}/{interface_type}", e.g.:
#   "left_joint1/position"  — joint position (radians)
#   "left_joint1/velocity"  — joint velocity (rad/s)
#   "left_joint1/effort"    — joint torque   (N·m)
#   "left_cam/rgb"          — camera RGB     (uint8 H×W×3)
#   "left_cam/depth"        — camera depth   (float32 H×W×1)
#   "left_end_effector/pose" — link pose (sapien.Pose)

InterfaceKey     = str
StateInterface   = dict[InterfaceKey, torch.Tensor]
CommandInterface = dict[InterfaceKey, torch.Tensor]
