---
name: weather-format
description: Format raw weather observations into a consistent human-readable daily report. Use when the user provides hourly observations, forecast tuples, or station readouts and asks for a clean summary.
---

When formatting a weather report, follow this template exactly. Treat each
field as required unless the input clearly does not contain the underlying
observation; in that case, write the literal string "n/a" rather than omitting
the field, so downstream parsers can rely on a fixed shape.

The report header is always three lines:

    Location: <city>, <region>
    Date:     <YYYY-MM-DD>
    Source:   <station id or provider>

After the header, leave one blank line and emit the conditions block. The
conditions block is a four-row aligned table whose left column is the
metric name padded to twelve characters and whose right column is the
formatted value. The four rows, in order, are temperature, humidity, wind,
and precipitation. Temperature is reported as the daily high and low joined
by a slash, both in degrees Celsius, no decimal places. Humidity is the
daytime mean as a percentage with no decimal. Wind is the prevailing wind
direction as a compass abbreviation (N, NE, E, SE, S, SW, W, NW) joined by a
space to the daily mean speed in kilometers per hour, rounded to one
decimal. Precipitation is total accumulation in millimeters across the
twenty-four-hour window, rounded to one decimal. If the underlying provider
reports zero precipitation, write "0.0 mm" rather than "trace" or "none";
this matters because the threshold for "trace" varies between providers and
the comparison code downstream expects the numeric form.

After the conditions block, leave one blank line and emit the narrative
paragraph. The narrative paragraph is two sentences, no more. The first
sentence summarizes the dominant condition for the day in plain English
("Mostly sunny with a brief afternoon shower." is a good shape). The second
sentence calls out anything anomalous compared to the seasonal normal for
the location, or, if nothing is anomalous, simply states that conditions
were typical for the season. Do not editorialize; do not include forecasts
for subsequent days; do not include warnings or advisories — those belong
in a separate alerts block that this template does not cover.

After the narrative paragraph, the report ends. Do not append a signature,
a generation timestamp, or a disclaimer. The intent of the template is to
be drop-in replaceable across stations and dates, and any trailing material
defeats that property. If the caller asks for a multi-day rollup, repeat
the entire template once per day, separated by a single blank line, and do
not introduce a wrapping header or footer.
