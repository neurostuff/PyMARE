# Pinned so the reference values are reproducible.
#
# Two things are pinned, because either one moving would move the numbers:
# the R version (the base image) and the robumeta version (a dated CRAN
# snapshot, since ``install.packages`` on its own tracks whatever CRAN
# happens to be serving that day).
FROM r-base:4.4.1
RUN R -e 'install.packages("robumeta", repos="https://packagemanager.posit.co/cran/2024-07-01")'
COPY run_robu.R /opt/run_robu.R
ENTRYPOINT ["Rscript", "/opt/run_robu.R"]
