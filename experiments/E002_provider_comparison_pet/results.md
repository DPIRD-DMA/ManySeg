| recipe | provider | n | params (M) | val mIoU | val boundary F1 | test mIoU | epoch time (s, contended) | t vs manyseg (val) | t vs manyseg (test) |
|---|---|---|---|---|---|---|---|---|---|
| fastai_like | fastai | 3 | 41.22 | 0.8226 ± 0.0013 | 0.7899 ± 0.0013 | 0.8251 ± 0.0005 | 28.7 | +1.10 | +9.64 |
| fastai_like | manyseg | 3 | 24.44 | 0.8216 ± 0.0009 | 0.7877 ± 0.0040 | 0.8220 ± 0.0003 | 11.1 |  |  |
| fastai_like | smp | 3 | 24.44 | 0.8218 ± 0.0009 | 0.7866 ± 0.0018 | 0.8230 ± 0.0007 | 11.5 | +0.27 | +2.15 |
| standard | fastai | 3 | 41.22 | 0.8143 ± 0.0022 | 0.7766 ± 0.0013 | 0.8180 ± 0.0001 | 28.0 | +0.52 | +6.83 |
| standard | manyseg | 3 | 24.44 | 0.8135 ± 0.0009 | 0.7740 ± 0.0032 | 0.8149 ± 0.0008 | 10.9 |  |  |
| standard | smp | 3 | 24.44 | 0.8134 ± 0.0025 | 0.7734 ± 0.0029 | 0.8155 ± 0.0009 | 11.2 | -0.08 | +0.88 |

recipe effect for manyseg: fastai_like - standard = +0.0080 (Welch t +10.5)

recipe effect for smp: fastai_like - standard = +0.0083 (Welch t +5.4)

recipe effect for fastai: fastai_like - standard = +0.0083 (Welch t +5.6)

encoder weight fingerprints across all runs: {'b89adf45880df6fa'}
