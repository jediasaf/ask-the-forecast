# Vision eval — claude-opus-5, effort high

_2026-09-28 · 10 charts generated from the same data the tools serve._

| Check | Result |
| --- | --- |
| Detected the planted error | 6/6 (100%) |
| Reported the real figure too | 6/6 (100%) |
| Stayed quiet on a clean chart | 4/4 (100%) |
| Passed everything | 10/10 (100%) |

## Cases

| Case | Tampered | Detected | Correct | No false alarm | Tools |
| --- | --- | --- | --- | --- | --- |
| `fa-clean` | no | yes | yes | yes | 4 |
| `fa-foods3` | yes | yes | yes | yes | 5 |
| `fa-hobbies2` | yes | yes | yes | yes | 3 |
| `naive-clean` | no | yes | yes | yes | 2 |
| `naive-household1` | yes | yes | yes | yes | 3 |
| `fva-clean` | no | yes | yes | yes | 3 |
| `fva-foods1` | yes | yes | yes | yes | 4 |
| `bias-clean` | no | yes | yes | yes | 2 |
| `bias-foods2` | yes | yes | yes | yes | 3 |
| `volume-foods3` | yes | yes | yes | yes | 2 |
