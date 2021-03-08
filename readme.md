# ClothSim

ClothSim is a GPU version cloth simulator. This project is still updating...

Input model and simulation parameters can be modified in `config/scene.json`.

## Dependencies

- CUDA 11.0 (cublas, cusparse, cusolver)
- Eigen 3.3.7 (\*)
- freeglut (\*)
- glew (\*)

(\*) means have been included in the code base.

## Methods

Cloth model:

- Membrane energy: StVK model

- Bending energy: [discrete quadratic bending model](http://www.cs.columbia.edu/cg/quadratic/)

Sparse matrix construction method is adapted from [CAMA](http://gamma.cs.unc.edu/CAMA/).

Implicit Euler integration -> nonlinear equation -> solved with Newton's method -> linear equation -> solved with LLT/CG

## TODO

* [ ] **definiteness fix for stretching energy hessian**
* [ ] **handle update and motion scripts**
* [ ] output and checkpoint
* [ ] **Neo-Hookean model**
* [ ] Projective Dynamics
* [ ] collision detection