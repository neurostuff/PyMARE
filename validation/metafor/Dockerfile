# Pinned so the reference values are reproducible.
#
# Two things are pinned, because either one moving would move the numbers: the R
# version (the base image) and the metafor version (a dated CRAN snapshot, since
# ``install.packages`` on its own tracks whatever CRAN happens to be serving
# that day). The same R pin as validation/robumeta and validation/clubsandwich,
# so the three reference images share a base layer.
FROM r-base:4.4.1
RUN R -e 'install.packages("metafor", repos="https://packagemanager.posit.co/cran/2024-07-01")'
# All three generators, with Rscript as the entrypoint rather than one of them,
# so regenerate.sh can run each in turn against one built image.
COPY run_metafor.R run_escalc.R run_permutest.R /opt/
ENTRYPOINT ["Rscript"]
