# nanoGPT PPL race results

Target PPL: **250.0** (fixed validation subset, same batch order and seed).

| Run | Params | Opt | Hit target? | Steps to target | Train seconds to target | Throughput (tok/s) | Final/Best PPL |
|---|---|---|---|---|---|---|---|
| `gpt2small-ffn-L0d0-adamw-lr6.0e-4` | 123.69M | adamw | yes | 600 | 159.1s | 123,598 | 237.6 |
| `gpt2small-ffn-L0d0-muon-lr1.5e-2` | 123.69M | muon | yes | 350 | 99.2s | 115,580 | 237.2 |
| `gpt2small-fng-L128d5-muon-lr1.5e-2` | 89.47M | muon | no | 500+ | not hit | 139,441 | 30877.4 |
| `gpt2small-fng-L16d5-adamw-lr3.0e-4` | 82.82M | adamw | no | 4500+ | not hit | 158,482 | 1443.8 |
| `gpt2small-fng-L16d5-muon-lr1.5e-2` | 82.24M | muon | no | 4500+ | not hit | 147,715 | 845.4 |
| `gpt2small-fng-L32d4-c1.0-muon-lr1.5e-2` | 82.98M | muon | no | 500+ | not hit | 146,072 | 38087.3 |
| `gpt2small-fng-L32d5-c1.0-muon-lr1.5e-2` | 83.27M | muon | no | 500+ | not hit | 146,309 | 57368.4 |
| `gpt2small-fng-L64d5-muon-lr1.5e-2` | 85.34M | muon | no | 500+ | not hit | 144,251 | 25661.4 |
| `gpt2small-fng-L8d5-adamw-lr6.0e-4` | 81.72M | adamw | no | 4500+ | not hit | 159,080 | 1108.2 |
| `gpt2small-fng-L8d5-muon-lr1.5e-2` | 81.72M | muon | no | 500+ | not hit | 147,831 | 2163.3 |
