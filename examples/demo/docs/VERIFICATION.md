# Numerical verification

Run `python -m heat_sim.verification --output results/verification.json` from the project environment. Add `--sensitivity` to run the representative laser mesh, time-step and domain-size comparison. The JSON records actual results, elapsed time, process resident memory, machine description, tested configuration, and any failed cases. A coarse/refined comparison is a resolution diagnostic, not proof that the continuum model is converged.

The verification runner exercises the same finite-volume face-flux operator used by the laser solver. For a homogeneous insulated slab it checks the known solution

`T(x,t) = 300 K + 20 K cos(pi x / L) exp[-alpha (pi/L)^2 t]`.

The solver uses cell-centered temperatures, so the analytical solution is evaluated at cell centers. Success requires small temperature error and decreasing error on mesh refinement. A second check integrates all face fluxes and checks that internal fluxes cancel to roundoff, including heterogeneous conductivities. Boundary cooling is checked against the explicitly tracked outward boundary heat flow.

The laser Gaussian check integrates the exact finite-cell flux weights on a domain many beam radii wide. The total approaches unity; multiplying by absorbed delivered power therefore gives watts. An offset beam near the boundary is also tested: missing beam power must leave the domain instead of being renormalized into it. The internal calculations use metres, seconds, kelvin, joules, watts and kilograms. The user preset 1,400 mm/s is 1.4 m/s and 0.06 mm is 60e-6 m.

Material enthalpy tests check monotonicity, temperature recovery and latent heat through the mushy interval. Sensible enthalpy plus latent heat is included in energy accounting. Powder cell mass stays fixed when its conductivity changes after melting; this avoids creating material or latent energy through the consolidation switch. The fixed-grid model does not simulate powder shrinkage.

For representative laser runs, the energy audit is `stored enthalpy change = absorbed source energy - outward boundary losses`, including convection/radiation and the fixed-temperature walls. The report includes the relative residual. An accurate energy audit alone does not establish accurate melt-pool geometry. The layer has three cells in coarse mode and six in refined mode; a depth change comparable to a cell is unresolved. Domain tests enlarge the thermal surroundings; a substantial geometry shift or heating at the remote boundaries requires a larger domain.

Failed simulations are recorded with their error text and are excluded from geometry comparisons and ranking. Numerical verification is separate from experimental validation. No comparison with a measured successful track is bundled: beam dimensions, effective absorptivity and powder conductivity remain unvalidated inputs. Flow, vaporization and recoil pressure are absent from the model, and convergence cannot correct those omissions.

## Recorded M4 demonstration

`examples/verification-m4.json` contains a run on the available Apple M4 MacBook Air (16 GiB physical RAM), using the explicitly assumed 80 µm beam, 0.35 absorptivity, 0.55 powder packing, 1 W/(m K) powder conductivity and 80 °C initial temperature. These assumptions are not printer measurements. Timings are observations from this process with the CPU kernels available in the local JIT cache; first-ever compilation adds overhead.

The cosine benchmark maximum error decreased from 0.002443 K (16 cells) to 0.000596 K (32 cells) to 0.000147 K (64 cells), consistent with second-order spatial behavior in this test. Gaussian integrated power, latent enthalpy, insulated flux conservation and zero-input cooling checks passed.

The initial demonstration predicted 116.76 µm width / 37.69 µm depth on the coarse mesh and 125.36 µm / 38.50 µm on the refined mesh. Depth is measured from the original powder top. Neither footprint reaches below the 60 µm powder layer. Width changed by 7.37%, so geometry convergence is **not established**. Halving the coarse time step changed width and depth by less than 0.04%. The energy audit closed to approximately 4e-15 relative in these runs.

**The peak temperatures (about 5,850 K coarse and 9,850 K refined) are far outside the physical range of this conduction model.** Omitted vaporization, flow and recoil would matter strongly. The temperature fields, footprint numbers and candidate rankings in this demonstration must not be used to choose printer settings. Good energy closure and a passing conduction benchmark do not make these physical predictions credible. Measure the delivered beam size/profile, delivered power and a successful baseline melt-pool cross-section before interpreting a transfer to 316L.

The saved JSON records actual per-case wall time and resident-memory observations. Timing varies with background load and cached compilation. It is not a guarantee of performance for other settings or a full build.

The domain-size check preserves cell spacing and grid alignment while enlarging the surrounding material. Width and depth were unchanged to roundoff between the 45×30×33 and 69×46×48 meshes; the outer-cell temperature rise fell from 0.015 K to 3.3e-8 K. This supports adequate remote boundary distance for this short, assumed single-track case only.

| Case | Cells | Recorded solver time | Sampled solver-process RSS peak |
|---|---:|---:|---:|
| Coarse | 44,550 | 0.164 s | 164.4 MiB |
| Coarse, half time step | 44,550 | 0.320 s | 165.1 MiB |
| Refined | 356,400 | 5.769 s | 185.4 MiB |
| Larger domain | 152,352 | 0.557 s | 180.3 MiB |

These RSS values include Python, loaded scientific libraries and allocated fields. They are process measurements, not an isolated allocation count for the solver. The larger-domain case runs after the refined case; allocator reuse affects its memory measurement.
