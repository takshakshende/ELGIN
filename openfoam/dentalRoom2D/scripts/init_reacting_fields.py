#!/usr/bin/env python3
"""
init_reacting_fields.py
=======================

Usage (called from Allrun):
    python3 scripts/init_reacting_fields.py <case_dir> <last_time>
"""

import os
import re
import sys

# ---------------------------------------------------------------------------
# Header template for new field files
# ---------------------------------------------------------------------------

HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | foam-extend: Open Source CFD                    |
|  \\\\    /   O peration     | Version:     4.1                                |
|   \\\\  /    A nd           | Web:         http://www.foam-extend.org         |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

FOOTER = "\n// ************************************************************************* //\n"

# All wall patches in the dental-room geometry (must match blockMeshDict)
WALL_PATCHES = (
    "floor", "ceiling", "leftWall", "rightWall",
    "dentistObstacle", "patientObstacle",
)

# ---------------------------------------------------------------------------
# Field writers
# ---------------------------------------------------------------------------

def write_p_absolute(path, loc):
    """Absolute pressure [Pa] for compressible reactingParcelFoam."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type zeroGradient; }}" for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj="p")
        + "// Absolute pressure [Pa] reset for reactingParcelFoam.\n"
        + "// simpleFoam wrote kinematic p* [m^2/s^2] -> overwritten here.\n\n"
        + "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        + "internalField   uniform 101325;\n\n"
        + "boundaryField\n{\n"
        + "    airInlet        { type zeroGradient; }\n"
        + "    airOutlet       { type fixedValue; value uniform 101325; }\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote absolute p     -> {path}")


def write_T(path, loc, T_amb=293.0):
    """Carrier-gas temperature field, room-temperature ambient."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type zeroGradient; }}" for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj="T")
        + "// Carrier temperature [K] - 20 C room ambient.\n\n"
        + "dimensions      [0 0 0 1 0 0 0];\n\n"
        + f"internalField   uniform {T_amb};\n\n"
        + "boundaryField\n{\n"
        + f"    airInlet        {{ type fixedValue; value uniform {T_amb}; }}\n"
        + f"    airOutlet       {{ type inletOutlet; inletValue uniform {T_amb}; "
        + f"value uniform {T_amb}; }}\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote T              -> {path}")


def write_species(path, loc, name, Y_amb):
    """Species mass fraction field at uniform ambient value."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type zeroGradient; }}" for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj=name)
        + f"// Mass fraction Y_{name} at 50%% RH, 20 C ambient.\n\n"
        + "dimensions      [0 0 0 0 0 0 0];\n\n"
        + f"internalField   uniform {Y_amb};\n\n"
        + "boundaryField\n{\n"
        + f"    airInlet        {{ type fixedValue; value uniform {Y_amb}; }}\n"
        + f"    airOutlet       {{ type inletOutlet; inletValue uniform {Y_amb}; "
        + f"value uniform {Y_amb}; }}\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote species {name:<4} -> {path}")


def write_mut(path, loc):
    """Compressible turbulent dynamic viscosity."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type mutkWallFunction; Cmu 0.09; kappa 0.41; "
        f"E 9.8; value uniform 0; }}"
        for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj="mut")
        + "// Compressible turbulent dynamic viscosity mu_t [kg/(m s)].\n\n"
        + "dimensions      [1 -1 -1 0 0 0 0];\n\n"
        + "internalField   uniform 0;\n\n"
        + "boundaryField\n{\n"
        + "    airInlet        { type calculated; value uniform 0; }\n"
        + "    airOutlet       { type calculated; value uniform 0; }\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote mut            -> {path}")


def write_alphat(path, loc):
    """Compressible turbulent thermal diffusivity (rho * alpha_t)."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type alphatWallFunction; Prt 0.85; value uniform 0; }}"
        for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj="alphat")
        + "// Compressible turbulent thermal diffusivity alpha_t.\n\n"
        + "dimensions      [1 -1 -1 0 0 0 0];\n\n"
        + "internalField   uniform 0;\n\n"
        + "boundaryField\n{\n"
        + "    airInlet        { type calculated; value uniform 0; }\n"
        + "    airOutlet       { type calculated; value uniform 0; }\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote alphat         -> {path}")


def write_G(path, loc):
    """Radiation incident heat flux (radiation is off; field still required)."""
    bc_walls = "\n".join(
        f"    {p:<16}{{ type zeroGradient; }}" for p in WALL_PATCHES
    )
    body = (
        HEADER.format(cls="volScalarField", loc=loc, obj="G")
        + "// Radiation field (radiation off in radiationProperties).\n\n"
        + "dimensions      [1 0 -3 0 0 0 0];\n\n"
        + "internalField   uniform 0;\n\n"
        + "boundaryField\n{\n"
        + "    airInlet        { type zeroGradient; }\n"
        + "    airOutlet       { type zeroGradient; }\n"
        + bc_walls + "\n"
        + "    frontAndBack    { type empty; }\n"
        + "}\n"
        + FOOTER
    )
    with open(path, "w") as f:
        f.write(body)
    print(f"  [OK] wrote G              -> {path}")


def patch_wall_functions(path, plain_name, compressible_name):
    """Replace plain wall-function names with compressible:: variants in-place."""
    if not os.path.exists(path):
        print(f"  [WARN] {path} not present, skipping wall-function patch.")
        return
    with open(path) as f:
        text = f.read()
    new_text, nsub = re.subn(
        rf"\btype\s+{plain_name}\b",
        f"type            {compressible_name}",
        text,
    )
    if nsub == 0:
        print(f"  [WARN] {path}: no '{plain_name}' patches found.")
        return
    with open(path, "w") as f:
        f.write(new_text)
    print(f"  [OK] patched {os.path.basename(path):<8} walls "
          f"({nsub}x): {plain_name} -> {compressible_name}")


def update_location_header(path, new_loc):
    """Rewrite the FoamFile 'location' line of an existing field file."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        text = f.read()
    new_text, nsub = re.subn(
        r'location\s+"[^"]*"\s*;',
        f'location    "{new_loc}";',
        text,
        count=1,
    )
    if nsub:
        with open(path, "w") as f:
            f.write(new_text)


def remove_kinematic_phi(dest):
    """
    simpleFoam writes phi with volumetric dimensions [0 3 -1] (m^3/s).
    reactingParcelFoam expects mass flux [1 0 -1] (kg/s) and rebuilds phi
    from linearInterpolate(rho*U) & mesh.Sf() if no phi is found in the
    starting time directory. Delete it to let the compressible solver
    create the correctly-dimensioned phi.
    """
    phi_path = os.path.join(dest, "phi")
    if os.path.isfile(phi_path):
        os.remove(phi_path)
        print(f"  [OK] removed kinematic phi (mass flux will be rebuilt by reactingParcelFoam)")
    else:
        print(f"  [INFO] no phi file at {phi_path} (nothing to remove)")


def rename_to_zero(case_dir, last_time):
    """
    simpleFoam treats SIMPLE iterations as pseudo-time, so 'latestTime' for
    the converged Phase-1 solution is e.g. 1665.  reactingParcelFoam reads
    endTime literally; if endTime=30 < 1665, the time loop runs zero steps.
    """
    import shutil

    src = os.path.join(case_dir, str(last_time))
    dst = os.path.join(case_dir, "0")
    if not os.path.isdir(src):
        print(f"  [WARN] cannot rename: source directory {src} not found.")
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)
    print(f"  [OK] renamed {src} -> {dst} "
          f"(reactingParcelFoam will start Phase 2 at t=0)")

    # Purge any stale precursor time directories (1500/, 1600/, ...).
    purged = []
    for entry in os.listdir(case_dir):
        full = os.path.join(case_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry == "0" or entry == "0.orig":
            continue
        try:
            t = float(entry)
        except ValueError:
            continue
        if t > 0:
            shutil.rmtree(full)
            purged.append(entry)
    if purged:
        print(f"  [OK] purged stale precursor time dirs: {', '.join(sorted(purged))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <case_dir> <last_time>")
        sys.exit(1)

    case_dir, last_time = sys.argv[1], sys.argv[2]
    dest = os.path.join(case_dir, last_time)
    if not os.path.isdir(dest):
        print(f"ERROR: time directory not found: {dest}")
        sys.exit(1)

    print(f"\n=== init_reacting_fields: {case_dir}  (time={last_time}) ===")

    write_p_absolute(os.path.join(dest, "p"), "0")

    patch_wall_functions(
        os.path.join(dest, "k"),
        "kqRWallFunction",
        "compressible::kqRWallFunction",
    )
    patch_wall_functions(
        os.path.join(dest, "omega"),
        "omegaWallFunction",
        "compressible::omegaWallFunction",
    )

    # Update the FoamFile.location header on k and omega (which were just
    # written by simpleFoam at time '<last_time>') so they read 'location "0"'
    # after the directory rename below.
    update_location_header(os.path.join(dest, "k"),     "0")
    update_location_header(os.path.join(dest, "omega"), "0")
    update_location_header(os.path.join(dest, "U"),     "0")
    update_location_header(os.path.join(dest, "nut"),   "0")

    # Use loc="0" inside the field headers so the FoamFile location matches
    # the renamed directory created at the end of this script.
    write_T(os.path.join(dest, "T"),     "0")
    write_mut(os.path.join(dest, "mut"), "0")
    write_alphat(os.path.join(dest, "alphat"), "0")
    write_species(os.path.join(dest, "H2O"), "0", "H2O", 0.007)
    write_species(os.path.join(dest, "N2"),  "0", "N2",  0.993)
    write_G(os.path.join(dest, "G"),     "0")

    remove_kinematic_phi(dest)

    rename_to_zero(case_dir, last_time)

    print(f"\n  Transition complete. reactingParcelFoam starts Phase 2 at t=0 "
          f"(was {last_time} from simpleFoam).\n")


if __name__ == "__main__":
    main()
