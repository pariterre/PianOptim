# PianOptim

An optimal control project to simulate piano

## Getting ready

### bioptim

Either the `bioptim` submodule is expected to appear in `dependencies/bioptim`. If the folder is empty, you should initialize it using git. 

Otherwise, `bioptim` can be installed using `conda` from the `conda-forge` channel. 

### Prepare for vscode

In the `.vscode` folder, copy-paste the `.env.default` file to `.env`. Adjust the separator if neede (":" for UNIX and ";" for Windows). 


## Things to discuss

### What "not using the trunk" actually means?

There are two ways to model this:
1. We can remove the degrees-of-freedom for the trunk. This means it won't be able to move, but the dynamic of the arms is concurently messed up. The reason is any movement of the arm, whatever the internal forces it creates, can be balanced out by this infinitely strong trunk. So it may try to transfer as much generalized forces as possible to the trunk
2. The second method is to constraint the trunk to have a velocity equals to zero at each nodes, effectively nullifying the movement of the trunk. This has the advantage of keeping the dynamic intact, but may be much harder to optimize

- Whatever what is decide, do we prescribe the pose actually held?

### What is the best way to model the press phase?

- Should we track a speed profile of the key from actual data?
- Should we track a force profile at the finger from actual data?
- Should we model the force from key? Using an exponential to simulate the bed of the key? Free time?
- Should we have a dynamic model of the sound? Artificial intelligence? Free time?

### Fingers are way too lights

- This may cause (and probably is causing) problems when inverting the dynamics (reason why forced to use COLLOCATION?)
- Is this relevent for the question we are trying to answer?
