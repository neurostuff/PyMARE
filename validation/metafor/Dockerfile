# Pinned so the reference values are reproducible.
#
# Two things are pinned, because either one moving would move the numbers: the R
# version (the base image) and the metafor version (a dated CRAN snapshot, since
# ``install.packages`` on its own tracks whatever CRAN happens to be serving
# that day). The same pins as validation/robumeta, so the two reference images
# share a base layer.
FROM r-base:4.4.1
RUN R -e 'install.packages("metafor", repos="https://packagemanager.posit.co/cran/2024-07-01")'
COPY run_metafor.R /opt/run_metafor.R
ENTRYPOINT ["Rscript", "/opt/run_metafor.R"]
