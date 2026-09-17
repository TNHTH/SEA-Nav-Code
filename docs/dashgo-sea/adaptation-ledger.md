# Adaptation ledger (Milestone A)

| Topic | Decision | Provenance |
| --- | --- | --- |
| Observation | 550D (55×10), keep vy/wx/wy | upstream fbce672c effective |
| Actor | New 2D DashGo actor; Go2 3-action path retained for regression only | scientific-contract §4.2 |
| CBF | Unicycle lookahead LSE; filter mean only; common-t halfspaces | §4.3 + A1 |
| Rewards | Table III weights × dt=0.02 once; omega=L2([wx,wy]) | §4.4 |
| Profiles | Exactly four leave-one-out | §5 |
| Platform | DashGo primitive candidate geometry in adapter assets | §5 |
| Train gate | `TRAIN_READY=False` until runtime gates pass | A3/A7/A9 |
| User method repo | Never opened / never copied | authorization |
