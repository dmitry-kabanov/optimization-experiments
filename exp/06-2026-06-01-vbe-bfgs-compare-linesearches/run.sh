export JAX_PLATFORMS=cpu

export OUTDIR=_output
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

mkdir -p ${OUTDIR}

SCIPY=scipy_optimize
OPTIM=optim_jl

# ---
# Python SciPy.optimize
python ${SCRIPT} --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-3 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-4 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-5 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

# ---
# Julia Optim.jl
python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --linesearch HagerZhang >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --linesearch HagerZhang >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --linesearch HagerZhang >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

# ---
# Julia Optim.jl using linesearch=StrongWolfe for fairer comparison with SciPy
python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

# ---
# Julia Optim.jl using linesearch=BackTracking for fairer comparison with SciPy
python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --linesearch BackTracking >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --linesearch BackTracking >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --linesearch BackTracking >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"
