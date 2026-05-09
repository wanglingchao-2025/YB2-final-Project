Step1: Use sam3.py to segement the figures to get the mask
Step2: Then we run vggt.py to get the  camera parameters and dense depth, and unproject the depth to obtain a scene-aligned point map.
Step3: Then we run sam3d.py to get the high-quality 3D model for each instance in an object-local coordinate system.
Step4: At last, we run aligh.py to register each SAM3D reconstruction result to a unified VGGT world coordinate system through a robust cross model alignment process.

Reference: We use different environment ro run the code, the  environment could be seen in the github of original paper:
SAM3:https://github.com/facebookresearch/sam3
VGGT:https://github.com/facebookresearch/vggt
SAM3D:https://github.com/facebookresearch/sam-3d-objects
