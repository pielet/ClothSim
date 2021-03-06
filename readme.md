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

- Bending energy: ~~[discrete quadratic bending model](http://www.cs.columbia.edu/cg/quadratic/)~~ [Cubic Shells](http://www.cs.columbia.edu/cg/pdfs/140-cubicShells-a4.pdf)

Sparse matrix construction method is adapted from [CAMA](http://gamma.cs.unc.edu/CAMA/).

Implicit Euler integration -> nonlinear equation -> solved with Newton's method -> linear equation -> solved with LLT/CG

Collision:

* support [dry fritional contact for PD](https://hal.inria.fr/hal-02563307v2/document)

## TODO

* [x] **definiteness fix for stretching energy hessian**
* [x] regularization for near singular hessian matrix
* [ ] replace `chol` with `cusolverRf` in Newton's method
* [x] **handle update and motion scripts**
* [ ] output and checkpoint
* [ ] **Neo-Hookean model**
* [x] **Projective Dynamics**
* [x] L-BFGS
* [x] collision detection & handling (implicit sphere & plane)
* [x] implicit geometry visualization
* [ ] spring collision (for Newton and L-BFGS)

## Problems

* The sheet rotates down much slower when use PD (large Young's modulus) -> Seems it's the drawback of PD itself, see [Stable Constrained Dynamics](https://hal.inria.fr/hal-01157835v2/document) and [WRAPD](https://www-users.cs.umn.edu/~brow2327/wrapd/).
* Denser mesh seems heavier (bug) -> add shape scale (edge length) to bending stiffness

