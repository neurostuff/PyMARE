# Pinned so the reference values are reproducible.
#
# Three things are pinned, because any one moving would move the numbers: the R
# version (the base image) and the metafor and clubSandwich versions (a dated
# CRAN snapshot, since ``install.packages`` on its own tracks whatever CRAN
# happens to be serving that day). metafor is here because clubSandwich has no
# model of its own: ``vcovCR`` and ``coef_test`` take a fitted ``rma.uni`` and
# replace its covariance, so the fit has to come from metafor.
#
# The same R pin as validation/metafor and validation/robumeta, so the three
# reference images share a base layer.
FROM r-base:4.4.1
RUN R -e 'install.packages(c("metafor", "clubSandwich"), repos="https://packagemanager.posit.co/cran/2024-07-01")'
COPY run_clubsandwich.R /opt/run_clubsandwich.R
ENTRYPOINT ["Rscript", "/opt/run_clubsandwich.R"]
