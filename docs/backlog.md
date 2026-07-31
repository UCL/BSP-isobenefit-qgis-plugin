# Backlog

Open design questions and deferred work, roughly ordered. Items move out of here
into releases once decided.

## 1. Pre-burn hopeless pockets into existing fabric

An open pocket enclosed by existing development that is smaller than the minimum
settlement size can never host a viable new cluster, whatever the run does. Such
pockets could be burned into existing fabric before the simulation (not marked
unbuildable, which would wrongly block walk routing), so the CA does not spend
its population budget on cells the absorption step will remove anyway. Post-run
absorption stays as the general rule; this only handles the provably hopeless
pockets, and the practical effect is a few cells per town, which is why it is
parked.
