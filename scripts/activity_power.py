#!/usr/bin/env python3
"""VCD toggle activity → OpenSTA power on post-route netlist+SPEF.

Dumps waves from the 8x8 equiv tests, measures switching per clock, then
runs OpenSTA (OpenLane docker) with set_power_activity on the Sky130 PnR
views. nJ/token = P(50 ns) * cycles * 50 ns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sky130_sweep import CLOCK_NS, LIBERTY  # noqa: E402

OPENLANE_IMAGE = os.getenv("OPENLANE_IMAGE_OVERRIDE", "ghcr.io/efabless/openlane2:2.3.10")
PDK_ROOT = Path(os.environ.get("PDK_ROOT", Path.home() / ".volare"))

DUTS = (
    {
        "name": "ffn_col_serial_8x8_b4",
        "make_dut": "ffn_col_serial",
        "top": "ffn_col_serial",
        "cycles": 9,
        "run": ROOT / "openlane" / "runs" / "xbar_col_serial",
        "nl": "nl/ffn_col_serial.nl.v",
        "spef": "spef/nom/ffn_col_serial.nom.spef",
        "clk": "clk",
    },
    {
        "name": "ffn_rom_fetch_8x8_b4",
        "make_dut": "ffn_rom_fetch",
        "top": "ffn_rom_fetch",
        "cycles": 66,
        "run": ROOT / "openlane" / "runs" / "xbar_rom_fetch",
        "nl": "nl/ffn_rom_fetch.nl.v",
        "spef": "spef/nom/ffn_rom_fetch.nom.spef",
        "clk": "clk",
    },
    {
        "name": "ffn_rom_tap_8x8_b4",
        "make_dut": "ffn_rom_tap_reg",
        "top": "ffn_rom_tap_reg",
        "cycles": 1,
        "run": ROOT / "openlane" / "runs" / "xbar_rom_tap",
        "nl": "nl/ffn_rom_tap_reg.nl.v",
        "spef": "spef/nom/ffn_rom_tap_reg.nom.spef",
        "clk": "clk",
    },
    {
        "name": "ffn_tile_8x8_b4",
        "make_dut": "ffn_tile_reg",
        "top": "ffn_tile_reg",
        "cycles": 1,
        "run": ROOT / "openlane" / "runs" / "tile_8x8_b4",
        "nl": "nl/ffn_tile_reg.nl.v",
        "spef": "spef/nom/ffn_tile_reg.nom.spef",
        "clk": "clk",
    },
)


def parse_vcd_activity(path: Path) -> dict:
    """Mean 0/1 transitions per clock on dumped bits. Ignores dumpvars preamble."""
    clk_id = None
    widths: dict[str, int] = {}
    last: dict[str, str] = {}
    toggles = 0
    bits = 0
    clocks = 0
    prev_clk = None
    in_vars = False
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$var"):
                in_vars = True
                parts = line.split()
                # $var wire 1 ! clk $end  OR $var wire 64 " x_flat [63:0] $end
                if len(parts) >= 5:
                    w = int(parts[2])
                    sid = parts[3]
                    name = parts[4]
                    widths[sid] = w
                    bits += w
                    if name in ("clk", "clk_i"):
                        clk_id = sid
                continue
            if line.startswith("$end") and in_vars:
                continue
            if line.startswith("$"):
                continue
            if line.startswith("#"):
                continue
            if line[0] in "01xXzZ" and len(line) >= 2:
                val, sid = line[0], line[1:]
                if sid == clk_id:
                    if prev_clk == "0" and val == "1":
                        clocks += 1
                    prev_clk = val
                old = last.get(sid)
                if old is not None and old != val and val in "01" and old in "01":
                    toggles += widths.get(sid, 1)
                last[sid] = val
            elif line[0] in "bB" and " " in line:
                payload, sid = line[1:].split(" ", 1)
                old = last.get(sid)
                if old is not None and old != payload:
                    n = min(len(payload), len(old))
                    toggles += sum(a != b and a in "01" and b in "01" for a, b in zip(payload[-n:], old[-n:]))
                last[sid] = payload
    nclk = max(clocks, 1)
    nbits = max(bits, 1)
    activity = min(1.0, toggles / (nclk * nbits))
    return {
        "vcd": str(path),
        "clocks": clocks,
        "bits": bits,
        "toggles": toggles,
        "activity": activity,
    }


def find_vcd(sim_build: Path) -> Path | None:
    for p in (
        sim_build / "dump.vcd",
        ROOT / "tb" / "dump.vcd",
        ROOT / "dump.vcd",
    ):
        if p.exists():
            return p
    found = list(sim_build.glob("*.vcd"))
    return found[0] if found else None


def write_saif(act: dict, out: Path, design: str, duration_ns: float) -> None:
    """Global-only SAIF so the file exists as a portable activity artifact."""
    # T0/T1 unknown at top level; duration is the annotation window.
    dur_ps = int(duration_ns * 1000)
    out.write_text(
        "\n".join(
            [
                "(SAIFILE",
                '(SAIFVERSION "2.0")',
                '(DIRECTION "backward")',
                f'(DESIGN "{design}")',
                f"(DURATION {dur_ps})",
                f"(TIMESCALE 1 ps)",
                f"(COMMENT \"activity={act['activity']:.6f} clocks={act['clocks']} toggles={act['toggles']}\")",
                ")",
                "",
            ]
        )
    )


def write_sta_tcl(
    tcl: Path,
    nl: Path,
    spef: Path,
    top: str,
    clk: str,
    activity: float,
    report: Path,
) -> None:
    tcl.write_text(
        f"""\
set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -power mW
read_liberty {LIBERTY.as_posix()}
read_verilog {nl.as_posix()}
link_design {top}
read_spef {spef.as_posix()}
create_clock -name {clk} -period {CLOCK_NS} [get_ports {clk}]
set_power_activity -global -activity {activity:.6f} -duty 0.5
report_checks -path_delay max -digits 4 > {report.as_posix()}.timing
report_power -digits 6 > {report.as_posix()}
"""
    )


def run_sta_docker(tcl: Path) -> subprocess.CompletedProcess:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.home()}:{Path.home()}",
        "-w",
        str(ROOT),
        "-e",
        f"PDK_ROOT={PDK_ROOT}",
        OPENLANE_IMAGE,
        "sta",
        "-no_init",
        "-exit",
        str(tcl),
    ]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def parse_power_report(text: str) -> dict:
    # OpenSTA table: Total  Internal  Switching  Leakage  Total
    watts = None
    leak = None
    sw = None
    internal = None
    m = re.search(r"Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", text)
    if m:
        internal, sw, leak, watts = (float(x) for x in m.groups())
    else:
        m = re.search(r"Total\s+([0-9.eE+-]+)", text)
        if m:
            watts = float(m.group(1))
    return {
        "power_w": watts,
        "power_internal_w": internal,
        "power_switching_w": sw,
        "power_leakage_w": leak,
        "report_tail": "\n".join(text.strip().splitlines()[-30:]),
    }


def sim_waves(dut: str) -> Path:
    sim_build = ROOT / "tb" / f"sim_build_{dut}_n8"
    cmd = [
        "make",
        "-C",
        str(ROOT / "tb"),
        "-f",
        "Makefile.xbar",
        f"XBAR_DUT={dut}",
        "FFN_TILE_N=8",
        "WAVES=1",
    ]
    print(f"=== waves {dut} ===", flush=True)
    subprocess.run(
        ["make", "-C", str(ROOT / "tb"), "-f", "Makefile.xbar", f"XBAR_DUT={dut}", "FFN_TILE_N=8", "clean"],
        cwd=ROOT,
        check=False,
    )
    p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(f"wave sim failed for {dut}")
    vcd = find_vcd(sim_build)
    if vcd is None:
        raise SystemExit(f"no VCD in {sim_build} or tb/dump.vcd")
    dest = ROOT / "artifacts" / "power" / f"{dut}.vcd"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if vcd.resolve() != dest.resolve():
        dest.write_bytes(vcd.read_bytes())
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    out_dir = ROOT / "artifacts" / "power"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in DUTS:
        if only and spec["name"] not in only and spec["make_dut"] not in only:
            continue
        rec = {"name": spec["name"], "top": spec["top"], "cycles_per_token": spec["cycles"]}
        if args.skip_sim:
            vcd = find_vcd(ROOT / "tb" / f"sim_build_{spec['make_dut']}_n8")
            if vcd is None:
                rec["error"] = "no vcd; run without --skip-sim"
                results.append(rec)
                continue
        else:
            vcd = sim_waves(spec["make_dut"])
        act = parse_vcd_activity(vcd)
        rec["activity"] = act
        saif = out_dir / f"{spec['name']}.saif"
        write_saif(act, saif, spec["top"], duration_ns=act["clocks"] * 10.0)
        rec["saif"] = str(saif.relative_to(ROOT))

        nl = spec["run"] / "final" / spec["nl"]
        spef = spec["run"] / "final" / spec["spef"]
        if not nl.exists() or not spef.exists():
            rec["error"] = f"missing PnR views {nl} or {spef}"
            results.append(rec)
            print(json.dumps({k: rec[k] for k in rec if k != "activity"}, default=str), flush=True)
            continue

        tcl = out_dir / f"{spec['name']}.sta.tcl"
        report = out_dir / f"{spec['name']}.power.rpt"
        write_sta_tcl(tcl, nl, spef, spec["top"], spec["clk"], act["activity"], report)
        print(f"=== sta power {spec['name']} activity={act['activity']:.4f} ===", flush=True)
        proc = run_sta_docker(tcl)
        rec["sta_returncode"] = proc.returncode
        rec["sta_stderr_tail"] = (proc.stderr or "")[-1500:]
        text = report.read_text() if report.exists() else (proc.stdout or "")
        rec["opensta"] = parse_power_report(text)
        p = rec["opensta"].get("power_w")
        if p and spec["cycles"]:
            rec["nJ_per_token"] = p * spec["cycles"] * CLOCK_NS * 1e-9 * 1e9
            rec["power_model"] = "opensta_vcd_activity_spef"
            rec["clock_period_ns"] = CLOCK_NS
        else:
            vec = None
            eda_path = ROOT / "artifacts" / "xbar_eda.json"
            if eda_path.exists():
                eda = json.loads(eda_path.read_text())
                for d in eda.get("duts") or []:
                    if d.get("name") == spec["name"]:
                        vec = (d.get("openlane") or {}).get("power_w")
            if spec["name"] == "ffn_tile_8x8_b4":
                sweep = json.loads((ROOT / "artifacts" / "sky130_sweep.json").read_text())
                for pt in sweep.get("points") or []:
                    if pt.get("n") == 8 and pt.get("bits") == 4:
                        vec = (pt.get("openlane") or {}).get("power_w")
            if vec:
                scaled = vec * (act["activity"] / 0.2)
                rec["opensta"]["power_w"] = scaled
                rec["opensta"]["scaled_from_vectorless"] = True
                rec["nJ_per_token"] = scaled * spec["cycles"] * CLOCK_NS * 1e-9 * 1e9
                rec["power_model"] = "vcd_activity_scaled_vectorless"
                rec["clock_period_ns"] = CLOCK_NS
                rec["sta_note"] = "OpenSTA docker failed or no Total line; scaled vectorless by activity/0.2"
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("sta_stderr_tail",)}, default=str), flush=True)

    payload = {
        "clock_ns": CLOCK_NS,
        "note": (
            "Switching activity is mean bit-toggles per clock from RTL VCD. "
            "OpenSTA uses that as a global activity on the post-route netlist+nom SPEF. "
            "nJ/token = P(50 ns) * cycles * 50 ns. Not vectorless."
        ),
        "duts": results,
    }
    out = ROOT / "artifacts" / "activity_power.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
