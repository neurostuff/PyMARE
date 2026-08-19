# Contributing to PyMARE

Welcome to the PyMARE repository! We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved. If you have any questions that aren't discussed below, please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in. Here are some [instructions][link_signupinstructions].

## Governance

Governance is a hugely important part of any project.
It is especially important to have clear process and communication channels for open source projects that rely on a distributed network of volunteers, such as ``PyMARE``.

`PyMARE` is currently supported by a small group of core developers.
Even with only a couple of individuals involved in decision making processes, we've found that setting expectations and communicating a shared vision has great value.

By starting the governance structure early in our development, we hope to welcome more people into the contributing team.
We are committed to continuing to update the governance structures as necessary.
Every member of the ``PyMARE`` community is encouraged to comment on these processes and suggest improvements.

All potential changes to ``PyMARE`` are explicitly and openly discussed in the described channels of communication, and we strive for consensus amongst all community members.

## Code of conduct

All ``PyMARE`` community members are expected to follow our [code of conduct](https://github.com/neurostuff/PyMARE/blob/master/CODE_OF_CONDUCT.md) during any interaction with the project.
That includes- but is not limited to- online conversations, in-person workshops or development sprints, and when giving talks about the software.

As stated in the code, severe or repeated violations by community members may result in exclusion from collective decision-making and rejection of future contributions to the ``PyMARE`` project.

## Labels

The current list of labels are [here][link_labels] and include:

* [![Good First Issue](https://img.shields.io/badge/-good%20first%20issue-7057ff.svg)](https://github.com/neurostuff/PyMARE/labels/good%20first%20issue)
*These issues contain a task that a member of the team has determined should require minimal knowledge of the existing codebase, and should be good for people new to the project.*
If you are interested in contributing to PyMARE, but aren't sure where to start, we encourage you to take a look at these issues in particular.

* [![Help Wanted](https://img.shields.io/badge/-help%20wanted-33aa3f.svg)](https://github.com/neurostuff/PyMARE/labels/help%20wanted)
*These issues contain a task that a member of the team has determined we need additional help with.*
If you feel that you can contribute to one of these issues, we especially encourage you to do so!

* [![Bug](https://img.shields.io/badge/-bug-ee0701.svg)](https://github.com/neurostuff/PyMARE/labels/bug)
*These issues point to problems in the project.*
If you find new a bug, please give as much detail as possible in your issue, including steps to recreate the error.
If you experience the same bug as one already listed, please add any additional information that you have as a comment.

* [![Enhancement](https://img.shields.io/badge/-enhancement-84b6eb.svg)](https://github.com/neurostuff/PyMARE/labels/enhancement)
*These issues are asking for new features to be added to the project.*
Please try to make sure that your requested feature is distinct from any others that have already been requested or implemented. If you find one that's similar but there are subtle differences please reference the other request in your issue.

## Making a change

We appreciate all contributions to PyMARE, but those accepted fastest will follow a workflow similar to the following:

**1. Comment on an existing issue or open a new issue referencing your addition.**

This allows other members of the PyMARE development team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front is so useful to everyone involved.

**2. Fork PyMARE.**

[Fork][link_fork] the [PyMARE repository][link_pymare] to your profile.

This is now your own unique copy of PyMARE. Changes here won't effect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date][link_updateupstreamwiki] with the master repository.

**3. Make the changes you've discussed.**

Try to keep the changes focused. We've found that working on a [new branch][link_branches] makes it easier to keep your changes targeted.

When you're creating your pull request, please do your best to follow PyMARE's preferred style conventions. Namely, documentation should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) convention and code should adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/) as much as possible.

**4. Submit a pull request.**

Submit a [pull request][link_pullrequest].

A member of the development team will review your changes to confirm that they can be merged into the main codebase.

## Running the tests

Install the test dependencies and run the suite:

```bash
pip install -e .[tests]
make unittest
```

The suite is organized one test file per source module -- `test_stats.py` covers
`pymare/stats.py`, `test_estimators.py` covers `pymare/estimators/estimators.py`,
and so on. Fixtures live in `pymare/tests/conftest.py` and helpers that are
neither fixtures nor tests live in `pymare/tests/utils.py`, so a test file holds
only tests.

Two groups of tests are marked, because they need something the default
environment does not have:

| Target | What it runs | Needs |
| --- | --- | --- |
| `make unittest` | everything except the Stan sampling tests | nothing extra |
| `make test_stan` | the Stan sampling tests | `pip install -e .[stan]`, then `make install_cmdstan` |
| `make test_robumeta` | the robumeta alignment tests | nothing extra |
| `make check_robumeta_alignment` | regenerates the robumeta reference values | Docker |
| `make lint` | flake8 over `pymare` and `benchmarks` | nothing extra |

Each of these has a GitHub Actions job behind it, so a target that passes
locally is the same check that runs on your pull request.

`make test_stan` needs two installation steps rather than one: the `stan` extra
brings in cmdstanpy, but CmdStan itself is a C++ build rather than a Python
package, so `make install_cmdstan` fetches and builds it. That takes several
minutes the first time and nothing thereafter.

**Those tests skip locally when CmdStan is missing, but fail in CI.** The
asymmetry is deliberate. A contributor without CmdStan should not see red, but a
skip is indistinguishable from a pass in a CI log, and that is exactly how the
Stan job passed for years while running none of the tests it existed to run --
its gate probed for a module name that PyStan 3 never provided. The Stan job now
sets `PYMARE_REQUIRE_CMDSTAN=1`, and the `pytest_collection_modifyitems` hook in
`pymare/tests/conftest.py` fails the run outright, at collection, wherever that
is set and CmdStan is missing.

Only the tests that actually sample are marked `stan`. The ones that check how
PyMARE's inputs are translated into Stan's data block need neither cmdstanpy nor
CmdStan, so they are unmarked and run in the ordinary unit job on every
platform.

The model's accuracy is measured separately, in `validation/stan/`, which
reports bias and credible-interval coverage across a grid of designs. That is
not run in CI; see its README for what it measured and how to rerun it.

### Alignment with robumeta

`pymare/tests/test_robumeta_alignment.py` pins PyMARE's correlated-effects model
against the R package [robumeta][link_robumeta], over every combination of model,
rho and variance column that both implementations can express. robumeta cannot be
a test dependency, so its output is pinned in
`pymare/tests/data/robumeta_reference.json`.

`make check_robumeta_alignment` regenerates that file inside a Docker image with
pinned R and robumeta versions, and fails if any number moved. The
`Check robumeta alignment` workflow runs the same script on every pull request,
so a change to the estimator that breaks agreement shows up as a failing check
rather than as a stale pin. If you changed the estimator on purpose, rerun the
script and commit the regenerated file.

### Benchmarks

Performance is guarded by [asv][link_asv]. The suite lives in `benchmarks/`, and
the `Benchmark` workflow times a pull request against its base branch and fails
if a benchmark is at least 1.3x slower with a statistically significant
difference. To run it once locally:

```bash
pip install asv virtualenv
asv machine --yes
make benchmark
```

To reproduce what CI does, compare two commits:

```bash
asv continuous --factor 1.3 --split master HEAD
```

The threshold is a tradeoff against measurement noise, so a flagged benchmark is
worth re-running before it is believed -- a loaded machine can move the cheapest
benchmarks by more than a real regression would. Re-running the workflow is
enough; `workflow_dispatch` also takes a different factor if you want one.

`benchmarks/bench_cluster_robust.py` is not part of the asv suite. It is a
standalone report on where the time in a robust fit goes; run it with
`python benchmarks/bench_cluster_robust.py`.

## Recognizing contributions

We welcome and recognize all contributions from documentation to testing to code development. You can see a list of current contributors in our [zenodo][link_zenodo] file. If you are new to the project, don't forget to add your name and affiliation there!

## Thank you!

You're awesome.

* NOTE: These guidelines are based on contributing guidelines from the [STEMMRoleModels][link_stemmrolemodels] project.

[link_github]: https://github.com/
[link_pymare]: https://github.com/neurostuff/PyMARE
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_react]: https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments
[link_issues]: https://github.com/neurostuff/PyMARE/issues
[link_labels]: https://github.com/neurostuff/PyMARE/labels
[link_discussingissues]: https://help.github.com/articles/discussing-projects-in-issues-and-pull-requests

[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_stemmrolemodels]: https://github.com/KirstieJane/STEMMRoleModels
[link_zenodo]: https://github.com/neurostuff/PyMARE/blob/master/.zenodo.json
[link_robumeta]: https://cran.r-project.org/package=robumeta
[link_asv]: https://asv.readthedocs.io/en/stable/
