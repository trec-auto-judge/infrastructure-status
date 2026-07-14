# TREC-AutoJudge Infrastructure Status Checks

Daily CI actions that run to monitor the status of the TREC AutoJudge infrastructure.

The infrastructure status checks run selected status checks of TREC AutoJudge twice per day (we can reduce that to maybe once every few days when the infrastructure is not actively needed). The [test-matrix.yml](test-matrix.yml) defines the workloads, for instance:
- Downloading selected PROMPT caches
- Running some AutoJudge systems from TIRA on the prompt caches, to ensure the evaluation yields the same outputs

We will further expand this.
