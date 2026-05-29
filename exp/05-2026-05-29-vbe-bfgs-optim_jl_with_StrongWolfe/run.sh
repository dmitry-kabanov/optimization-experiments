export JAX_PLATFORMS=cpu

export OUTDIR=_output
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

mkdir -p ${OUTDIR}

OPTIM=optim_jl


# ---
# Julia Optim.jl

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python ${SCRIPT} --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --linesearch StrongWolfe >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"
