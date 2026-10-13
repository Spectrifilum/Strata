# A/B speed sweep for this PC (RX 9070 16 GB, gfx1201, Windows, Q2_0)

`bench/ab_speed.py` starts its own engine per variant and measures it. It never touches a
server that is already running, and it refuses to start while the card is busy.

## Run it

```bat
:: is the card free?
python bench/ab_speed.py --check-only

:: quick baseline of the config you use every day, no restart, no risk
python bench/ab_speed.py --live

:: the promising switches, one engine start each (~15-20 min)
python bench/ab_speed.py

:: the two arms that matter, interleaved A/B/A/B (the method the docs use)
python bench/ab_speed.py --pairs baseline pcie-auto --rounds 2

:: everything on the list
python bench/ab_speed.py --all --rounds 2
```

Before the sweep (not for `--live`): unload the model you are using - the unload button on
`http://127.0.0.1:12345`, or Ctrl+C on the server window. One card holds one engine.

Results go to `bench/ab/<date>-rx9070-sweep/`: `summary.md` (the table) and `results.json`
(every request, plus the engine's start-up lines: expert cache slots, VRAM free, PCIe share,
tail skip). Send me `summary.md` and I will read it with you.

## What each arm changes, and why it is on the list

| arm | change | where the idea comes from |
| --- | --- | --- |
| `pcie-auto` | drop `--pcie-frac 0.00`, let the engine probe the link and refine the share | docs/DETAILS.md "The PCIe share of the missed experts is set from the CPU's speed": on an RTX 3060 the measured share was 1.30x the old default; the engine's own refinement is what `--pcie-frac 0.00` switches off |
| `pcie-0.25` | a fixed quarter of the missed experts on the GPU | the middle of the range the docs report for cards like this |
| `draft-en` | `--draft-vocab en` | docs/DETAILS.md: the default draft vocabulary carries 106,299 CJK ids (~180 MiB of VRAM); the English/code subset is ~110 MiB smaller and English answers 1-2% faster |
| `cache-every-4096` | `--prompt-cache-every 4096` | docs/DETAILS.md "Conversation cache": checkpoints are taken every 16K tokens, so a follow-up turn re-reads up to 16K tokens of the chat |
| `conv-cache` | `--conversation-cache-mib 8192 --conversation-cache-slots 4` | docs/DETAILS.md "Multiple conversations": parking is off by default (budget 0) |
| `select-wmma` | `STRATA_SELECT_WMMA=1` | docs/AMD_HIP.md RDNA4 section: +1.5% on 16K prompts at a 262K context on gfx12 - this machine runs 262K |
| `prefill-16k` | `STRATA_PREFILL_AUTO_MAX=16384` | docs/DETAILS.md: gfx12 may take 16384-token chunks, but only on Linux today (+0.5..+5.8% on an R9700) |
| `host-core-last` | `--host-core last` | docs/DETAILS.md 0.1.40 switches: on Windows the GPU's interrupts land on core 0, where the host thread spins |
| `reserve-1000` | `--vram-reserve-mib 1000` | docs/DETAILS.md "The expert cache size is a budget": slack on the card beat extra slots by a wide margin (13.7 vs 102 tok/s on a 4080 SUPER) |
| `adapt-overlap` | `STRATA_ADAPT_OVERLAP=1` | docs/DETAILS.md: +5.9% decode on an A4000; opt-in on Windows/HIP |
| `adapt-async` | `--adapt-async 1` | docs/DETAILS.md: cache swaps advance between windows on a helper thread |
| `no-pcie-group` | `STRATA_NO_PCIE_GROUP=1` | PR #2223 - not in the 0.1.43 engine; only means something if you build the engine with that PR |
| `hc-persist` | `STRATA_HC_PERSIST=1` | PR #2226 - same caveat |

## Things already on, so do not chase them

- `STRATA_HIP_WMMA=1` is in your config: the RDNA4 matrix-core prompt attention (R9700: 1,784 -> 2,427 tok/s at 4K). With it the engine already picks the 96-slot prompt ring that was measured better on a 9070 XT.
- The gfx1201 hipBLASLt table matches the shipped library (`prefill gemm: hipBLASLt tuning enabled (32 rows, gfx1201, version 100500)`).
- `STRATA_ROUTE_TAIL_SKIP` and the measured PCIe share default are on for gfx1200/gfx1201 since 0.1.43 - but only when no `--pcie-frac` is given, which is why `pcie-auto` is the first arm to try.
- `STRATA_PF_FUSED=1` does nothing for the Q2_0 pack on gfx12 (docs/AMD_HIP.md: "the Q2_0 pack does not take these kernels on gfx12").

## Reading the table

Compare a variant to `baseline` in the same column. A real win is more than one round of
noise: run `--pairs A B --rounds 2` and look at whether the same arm wins in both rounds.
The `cache slots` and `VRAM free` columns explain the speed: if a variant grows the cache and
VRAM free falls under 256 MiB, the engine is paging, and the slots are not a win.
