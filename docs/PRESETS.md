# Print preset comparisons

Use **Preset lab** to compare changes to an imported print before trying them on the machine. Presets are named physical settings. The coarse/preview choices in **Calculate** control numerical scope and resolution separately.

1. Set the current material, independent optical inputs, and machine assumptions in **Materials & settings**.
2. In **Calculate**, choose the whole print, a quick preview, or a custom layer range. Every case in a preset comparison uses this same source geometry, range, thermal grid, and frame settings.
3. In **Preset lab**, name an alternative and edit its material, power/speed percentages, preheat, inter-layer pause, beam diameter, substrate, or powder and jump inputs. Save it. To branch an existing preset, change its name and use **Save as new preset**.
4. Choose **Current job settings** or a saved successful preset as the reference. Select up to twelve saved candidates and choose **Simulate preset comparison**. The summary lists the actual frozen settings, conditional physical time, and step preflight for every case. **Live simulation** opens automatically and refreshes actual completed thermal frames while the worker calculates. Preparing timing or a case that has not saved its first field shows a waiting state; no temperatures are invented.
5. After completion, use **Interpret these results**. **Results insights** shows the selected study, closest sampled thermal response, changes from the reference, actual peak temperature, retained domain heat, physical duration and sampled temperature histories. **Adjust this preset** opens a draft revision in Preset lab. Save and simulate it to continue tuning.

## Region recipes

Expand **Tune individual region recipes** to replace infill, contour, support or other source-section settings independently. Enable each power or speed replacement explicitly. Unchecked fields keep every vector's imported value. Checked fields use the specified absolute W or mm/s for every part sharing that section.

The global power and speed percentages apply **after** region replacements. For example, section 6 power of 100 W with a global power setting of 110% gives a 110 W command. A shared delivered-power factor of 0.9 then gives 99 W delivered in the thermal model. A section speed of 1,000 mm/s with 90% speed gives 900 mm/s. The effective-region table distinguishes command power from assumed delivered power.

The importer’s W and mm/s interpretation remains provisional until verified for the printer. Geometry, vector order, and original job files are preserved. Changing hatch spacing, layer thickness, scan strategy or path geometry requires a newly sliced/imported job.

## Reading the results

The score is a relative RMS difference between reference and candidate mean, 95th-percentile and maximum peak enthalpy indices over exposed part cells. Its denominator floor is 0.02. A lower score means a closer **sampled resolved heat response**. It does not measure defect probability, density or printing success. Each material and preheat uses its own enthalpy normalization, so compare absolute temperatures and retained heat as well when changing preheat.

Cases with failed integration, a missing reference, or different actual exposed path/grid/voxel footprints are excluded from ranking. Time limits can make different-speed cases incomparable. All attempts and completed partial frames remain identifiable. Duplicate named physical settings may share computed artifacts while keeping distinct names.

Retained heat is excess energy in the entire simulated part, powder and plate at the end of the selected range and cooldown. Temperature histories use actual stored samples. Short peaks can be missed between saved frames. Coarse cell averages do not establish microscopic melt-pool width/depth or keyhole behavior. Small score differences need numerical refinement and measurement before narrowing physical trials.

## Watching a calculation or finished print

**Live simulation** follows the current calculation's latest saved frame about every two seconds. It identifies the run and active material/preset case. The final available frame stays visible after completion, cancellation or failure; its status explains whether the calculation finished.

**Build playback → Full simulated print** replays the entire saved timeline using all saved thermal frames. The overview is accelerated for long print schedules. **Show finished print** jumps to the final saved state; **Open latest full print** selects a verified full-source result rather than a newer short preview. The separate **One layer: detailed scan paths** mode intentionally restricts playback to one layer's detailed event clock. Choose the full mode again to watch the complete history. Powder starts hidden to make the exposed part easier to see.

Live refresh and replay both use actual converged saved temperatures. The full overview of a large schedule cannot show an exact beam position between frames unless that detailed event scope is embedded. Layer timing comes from the saved schedule, while the thermal field is held at the latest stored frame. A coarse thermal grid can show a blocky part, and a run calculated over only a few layers cannot supply temperatures for the rest of the print.

## Saved settings

Libraries live locally under `results/presets/`, scoped to the imported job hash. Each simulation stores complete reference/candidate snapshots and numerical source under its own `results/builds/` folder. Later library edits leave those results unchanged. The library can be exported as JSON; study comparison CSV and a plain-language interpretation can be downloaded from their result views. No preset is sent to a printer.
