# Simulation: Generation / Editing

- Generative models for synthetic data: diffusion model, Gaussian Splatting / neural rendering

- Sim2Real and Real2Sim: Common in robotics for transferring policies or data between simulation and the real world

- Scene-level generation / editing: [ScanEdit](https://arxiv.org/abs/2504.15049) takes a scanned 3D scene with object instances and uses an LLM to interpret high level editing instructions such as move or arrange to produce a rearranged scene.

- Object-level synthetic generation / editing: [MeshCoder](https://arxiv.org/abs/2508.14879) takes a 3D point cloud of an object and uses a multimodal LLM to output a Blender Python script that reconstructs the object and allows editing through code. [MeshPad](https://arxiv.org/abs/2503.01425) provides an interactive tool for 3D mesh creation and editing driven by 2D sketches.

- Procedural 3D scene or dataset generation: for example Unity and Blender based pipelines
