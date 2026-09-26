from datetime import datetime
from itertools import combinations
from pathlib import Path
import json
import re


project_folder = Path(__file__).resolve().parent.parent
logs_folder = project_folder / "logs"
data_folder = project_folder / "data"
output = data_folder / "prompts.jsonl"

data_folder.mkdir(exist_ok=True)

number = r"-?\d+(?:\.\d+)?"

appliance_pattern = re.compile(
    rf"^([A-Za-z][A-Za-z0-9_]*)\s+[\d,]+\s+"
    rf"({number})\s+({number})\s+({number})\s+({number})\s*$",
    flags=re.MULTILINE,
)

signal_pattern = re.compile(
    r"^\s*-- Signal variant:.*?--\s*$",
    flags=re.MULTILINE,
)

aggregate_header = re.compile(
    r"^\s*\[([^\]]+)\]\s+Disaggregation Metrics\s+\[[^\]]+\]",
    flags=re.MULTILINE,
)

metric_patterns = {
    "accuracy": rf"Exact-match Accuracy\s+({number})",
    "macro_f1": rf"Macro F1 Score\s+({number})",
    "teca": rf"TECA \(Total Energy Correctly\)\s+({number})",
    "event_f1": rf"Event-based F1\s+({number})",
}

device_header = re.compile(
    r"^Device\s+MAE\s+RMSE\s+R2\s+ON-ACC\s+ON-F1\s*$",
    flags=re.MULTILINE,
)

device_row = re.compile(
    rf"^\s*([A-Za-z][A-Za-z0-9_]*)\s+"
    rf"({number})\s+{number}\s+({number})\s+{number}\s+({number})\s*$",
    flags=re.MULTILINE,
)

method_marker = re.compile(
    r"\[([^\]]+)\]\s+(?:Training\b|Loaded cached results)"
    r"|^\s*([A-Za-z0-9][A-Za-z0-9_-]*)\s+Seq2Point Analysis",
    flags=re.MULTILINE,
)

event_row = re.compile(
    rf"^\s*\d+\s*\|\s*(2019-\d\d-\d\d \d\d:\d\d:\d\d)\s*\|"
    rf"\s*(-?\d+)\s*\|\s*(-?\d+)\s*\|\s*({number})\s*\|\s*([+-])\s*$",
    flags=re.MULTILINE,
)


def split_blocks(text):
    markers = list(signal_pattern.finditer(text))
    blocks = []

    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        blocks.append(text[start:end])

    return blocks


def extract_aggregate(block):
    headers = list(aggregate_header.finditer(block))
    results = []

    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(block)
        section = block[start:end]
        result = {}

        for name, pattern in metric_patterns.items():
            match = re.search(pattern, section)
            if match:
                result[name] = float(match.group(1))

        if len(result) == 4:
            results.append(result)

    return results


def extract_devices(block):
    headers = list(device_header.finditer(block))
    results = []

    for index, header in enumerate(headers):
        markers = list(method_marker.finditer(block[:header.start()]))
        if not markers:
            continue

        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(block)
        devices = []

        for row in device_row.finditer(block[start:end]):
            devices.append({
                "name": row.group(1),
                "mae": float(row.group(2)),
                "r2": float(row.group(3)),
                "on_f1": float(row.group(4)),
            })

        if len(devices) == 10:
            results.append({"devices": devices})

    return results


def unique(items):
    result = []
    seen = set()

    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def choose_spread(items, score, count):
    ordered = sorted(items, key=score)
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[position] for position in positions]


log_files = sorted(logs_folder.glob("*.log"))

if not log_files:
    raise ValueError(f"No log files were found in {logs_folder}.")

texts = [
    path.read_text(encoding="utf-8", errors="replace")
    for path in log_files
]


appliances = {}
for match in appliance_pattern.finditer(texts[0]):
    appliances[match.group(1)] = {
        "mean": float(match.group(2)),
        "std": float(match.group(3)),
        "median": float(match.group(4)),
        "maximum": float(match.group(5)),
    }

if len(appliances) != 10:
    raise ValueError(f"Expected 10 appliances, found {len(appliances)}.")


aggregate_results = []
device_results = []
events = set()

for text in texts:
    for block in split_blocks(text):
        aggregate_results.extend(extract_aggregate(block))
        device_results.extend(extract_devices(block))

    for match in event_row.finditer(text):
        timestamp, old_state, new_state, power, direction = match.groups()
        if timestamp >= "2019-01-29":
            events.add((timestamp, old_state, new_state, float(power), direction))

aggregate_results = unique(aggregate_results)
device_results = unique(device_results)


def name(appliance):
    return appliance.replace("_", " ")


def variability_summary(values):
    mean = values["mean"]
    std = values["std"]
    median = values["median"]
    maximum = values["maximum"]

    if median == 0 and maximum > 5 * max(mean, 0.001):
        return "intermittent use with occasional high peaks"
    if std > mean:
        return "a highly variable power pattern"
    return "a relatively stable power pattern"


def confidence_level(values):
    if all(value >= 0.70 for value in values):
        return "high"
    if all(value < 0.50 for value in values):
        return "low"
    if all(0.50 <= value < 0.70 for value in values):
        return "moderate"
    return "mixed"


def reliability_level(r2, on_f1):
    if r2 >= 0.50 and on_f1 >= 0.70:
        return "high"
    if r2 < 0 or on_f1 < 0.30:
        return "low"
    return "moderate"


prompts = []
reply_rule = (
    "Give a short logically reasoned answer based on the provided information. "
    "Do not simply repeat the prompt or add unsupported claims."
)

all_appliance_names = list(appliances)

peak_groups = choose_spread(
    list(combinations(all_appliance_names, 3)),
    score=lambda group: sum(
        appliances[item]["maximum"] for item in group
    ),
    count=8,
)

for group in peak_groups:
    group_names = ", ".join(name(item) for item in group)
    highest = max(group, key=lambda item: appliances[item]["maximum"])
    highest_value = appliances[highest]["maximum"]
    prompt = (
        f"After comparing {group_names}, my monitor shows that the {name(highest)} "
        f"had the highest peak at {highest_value:.2f} VA. Give one safe recommendation "
        f"for managing it, "
        f"focusing on checking whether the peak repeats before changing its use. "
        f"{reply_rule}"
    )
    prompts.append(prompt)


comparison_pairs = choose_spread(
    list(combinations(all_appliance_names, 2)),
    score=lambda pair: abs(
        appliances[pair[0]]["maximum"]
        - appliances[pair[1]]["maximum"]
    ),
    count=8,
)

for first, second in comparison_pairs:
    higher_mean = max(
        (first, second),
        key=lambda item: appliances[item]["mean"],
    )
    higher_maximum = max(
        (first, second),
        key=lambda item: appliances[item]["maximum"],
    )
    prompt = (
        f"When comparing the {name(first)} and {name(second)}, my monitor shows that "
        f"the {name(higher_mean)} has the higher average "
        f"reading, while the {name(higher_maximum)} has the higher peak. Give one "
        f"practical recommendation for managing household demand based on this "
        f"difference. {reply_rule}"
    )
    prompts.append(prompt)


variable_appliances = sorted(
    all_appliance_names,
    key=lambda item: (
        appliances[item]["std"]
        / max(appliances[item]["mean"], 0.001)
    ),
    reverse=True,
)[:8]

for appliance in variable_appliances:
    values = appliances[appliance]
    pattern = variability_summary(values)
    prompt = (
        f"My monitor suggests that the {name(appliance)} has {pattern}, reaching "
        f"{values['maximum']:.2f} VA at its highest point. Recommend one safe next "
        f"step for monitoring this pattern before changing how the appliance is "
        f"used. {reply_rule}"
    )
    prompts.append(prompt)


for result in choose_spread(
    aggregate_results,
    score=lambda item: item["macro_f1"],
    count=6,
):
    confidence = confidence_level([
        result["accuracy"],
        result["macro_f1"],
        result["teca"],
        result["event_f1"],
    ])
    prompt = (
        f"This NILM result has a Macro F1 score of {result['macro_f1']:.4f} and a "
        f"{confidence} overall confidence level. Give one cautious recommendation "
        f"about using this result for a "
        f"household energy decision. {reply_rule}"
    )
    prompts.append(prompt)


def average_on_f1(result):
    return sum(item["on_f1"] for item in result["devices"]) / len(result["devices"])


for result in choose_spread(device_results, average_on_f1, 6):
    ordered = sorted(
        result["devices"],
        key=lambda item: (item["on_f1"], item["r2"]),
    )
    weaker = ordered[0]
    stronger = ordered[-1]

    strong_level = reliability_level(stronger["r2"], stronger["on_f1"])
    weak_level = reliability_level(weaker["r2"], weaker["on_f1"])

    if strong_level == "low" and weak_level == "low":
        prompt = (
            f"The {name(stronger['name'])} estimate (R2 {stronger['r2']:.3f}, "
            f"ON-F1 {stronger['on_f1']:.3f}) and the {name(weaker['name'])} estimate "
            f"(R2 {weaker['r2']:.3f}, ON-F1 {weaker['on_f1']:.3f}) both have low "
            f"reliability. Give one cautious recommendation for "
            f"verifying them before taking action. {reply_rule}"
        )
    else:
        prompt = (
            f"The {name(stronger['name'])} estimate (R2 {stronger['r2']:.3f}, "
            f"ON-F1 {stronger['on_f1']:.3f}) is more reliable than the "
            f"{name(weaker['name'])} estimate (R2 {weaker['r2']:.3f}, ON-F1 "
            f"{weaker['on_f1']:.3f}). Give one "
            f"cautious recommendation about which estimate to use. {reply_rule}"
        )
    prompts.append(prompt)


event_choices = []
used_hours = set()

for event in sorted(events, key=lambda item: abs(item[3]), reverse=True):
    hour = event[0][11:13]

    if hour not in used_hours:
        used_hours.add(hour)
        event_choices.append(event)

    if len(event_choices) == 4:
        break

if len(event_choices) < 4:
    raise ValueError("Not enough different event excerpts were found.")

for timestamp, _, _, power, direction in event_choices:

    direction_text = "upward" if direction == "+" else "downward"
    readable_time = datetime.strptime(
        timestamp,
        "%Y-%m-%d %H:%M:%S",
    ).strftime("%H:%M on %B %d, %Y").replace(" 0", " ")
    prompt = (
        f"My monitor detected an {direction_text} event of {power:.1f} VA at "
        f"{readable_time}, but it did not identify the appliance. Give one sensible "
        f"next step for investigating the event without guessing its source. "
        f"{reply_rule}"
    )
    prompts.append(prompt)


if len(prompts) != 40:
    raise ValueError(f"Expected 40 prompts, created {len(prompts)}.")

if len(set(prompts)) != 40:
    raise ValueError("Duplicate natural prompts were created.")

records = [
    {"id": f"N{index:03d}", "prompt": prompt}
    for index, prompt in enumerate(prompts, start=1)
]

with output.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {len(records)} prompts in {output}.")