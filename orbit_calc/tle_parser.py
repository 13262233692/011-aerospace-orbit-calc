import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TLEEntry:
    satellite_name: str
    norad_id: int
    classification: str
    international_designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_dot: float
    mean_motion_ddot: float
    bstar: float
    inclination: float
    raan: float
    eccentricity: float
    argument_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    revolution_number: int
    line1: str
    line2: str

    def to_lines(self) -> Tuple[str, str]:
        return (self.line1, self.line2)


def _parse_power_of_ten(val_str: str) -> float:
    val_str = val_str.strip()
    if not val_str:
        return 0.0

    match = re.match(r'^([+-]?)(\d+)([+-]\d+)$', val_str.replace(' ', ''))
    if match:
        sign = -1.0 if match.group(1) == '-' else 1.0
        mantissa = int(match.group(2)) / 10.0 ** len(match.group(2))
        exp_sign = -1 if match.group(3)[0] == '-' else 1
        exponent = int(match.group(3)[1:])
        return sign * mantissa * 10.0 ** (exp_sign * exponent)

    try:
        return float(val_str)
    except ValueError:
        return 0.0


def _validate_checksum(line: str) -> bool:
    if len(line) < 69:
        return False
    checksum = 0
    for ch in line[:-1]:
        if ch.isdigit():
            checksum += int(ch)
        elif ch == '-':
            checksum += 1
    return checksum % 10 == int(line[-1])


def parse_tle_line1(line1: str) -> dict:
    if len(line1) < 69:
        raise ValueError(f"Line 1 too short: {len(line1)} chars, expected 69")

    if line1[0] != '1':
        raise ValueError(f"Line 1 must start with '1'")

    return {
        'line_number': 1,
        'norad_id': int(line1[2:7].strip()),
        'classification': line1[7],
        'international_designator': line1[9:17].strip(),
        'epoch_year': int(line1[18:20]),
        'epoch_day': float(line1[20:32]),
        'mean_motion_dot': float(line1[33:43]),
        'mean_motion_ddot': _parse_power_of_ten(line1[44:52]),
        'bstar': _parse_power_of_ten(line1[53:61]),
        'ephemeris_type': int(line1[62]),
        'element_number': int(line1[64:68].strip()),
    }


def parse_tle_line2(line2: str) -> dict:
    if len(line2) < 69:
        raise ValueError(f"Line 2 too short: {len(line2)} chars, expected 69")

    if line2[0] != '2':
        raise ValueError(f"Line 2 must start with '2'")

    return {
        'line_number': 2,
        'norad_id': int(line2[2:7].strip()),
        'inclination': float(line2[8:16]),
        'raan': float(line2[17:25]),
        'eccentricity': int(line2[26:33]) * 1e-7,
        'argument_of_perigee': float(line2[34:42]),
        'mean_anomaly': float(line2[43:51]),
        'mean_motion': float(line2[52:63]),
        'revolution_number': int(line2[63:68].strip()),
    }


def parse_tle_entry(name: str, line1: str, line2: str) -> TLEEntry:
    l1 = parse_tle_line1(line1.strip())
    l2 = parse_tle_line2(line2.strip())

    if l1['norad_id'] != l2['norad_id']:
        raise ValueError(
            f"NORAD ID mismatch: line1={l1['norad_id']}, line2={l2['norad_id']}"
        )

    return TLEEntry(
        satellite_name=name.strip(),
        norad_id=l1['norad_id'],
        classification=l1['classification'],
        international_designator=l1['international_designator'],
        epoch_year=l1['epoch_year'],
        epoch_day=l1['epoch_day'],
        mean_motion_dot=l1['mean_motion_dot'],
        mean_motion_ddot=l1['mean_motion_ddot'],
        bstar=l1['bstar'],
        inclination=l2['inclination'],
        raan=l2['raan'],
        eccentricity=l2['eccentricity'],
        argument_of_perigee=l2['argument_of_perigee'],
        mean_anomaly=l2['mean_anomaly'],
        mean_motion=l2['mean_motion'],
        revolution_number=l2['revolution_number'],
        line1=line1.strip(),
        line2=line2.strip(),
    )


def parse_tle_file(content: str) -> List[TLEEntry]:
    lines = [l for l in content.strip().splitlines() if l.strip()]
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('1') and len(line) >= 69:
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('2') and len(next_line) >= 69:
                    try:
                        entry = parse_tle_entry("UNKNOWN", line, next_line)
                        entries.append(entry)
                    except ValueError:
                        pass
                    i += 2
                    continue
            i += 1
        elif line.startswith('2'):
            i += 1
        else:
            name = line
            if i + 2 < len(lines):
                l1 = lines[i + 1].strip()
                l2 = lines[i + 2].strip()
                if l1.startswith('1') and l2.startswith('2'):
                    try:
                        entry = parse_tle_entry(name, l1, l2)
                        entries.append(entry)
                    except ValueError:
                        pass
                    i += 3
                    continue
            i += 1

    return entries


def parse_tle_kafka_message(message_value: bytes) -> Optional[TLEEntry]:
    try:
        text = message_value.decode('utf-8') if isinstance(message_value, bytes) else message_value
    except UnicodeDecodeError:
        return None

    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 2:
        l1, l2 = lines[0].strip(), lines[1].strip()
        if l1.startswith('1') and l2.startswith('2'):
            try:
                return parse_tle_entry("UNKNOWN", l1, l2)
            except ValueError:
                return None
    elif len(lines) >= 3:
        name = lines[0].strip()
        l1, l2 = lines[1].strip(), lines[2].strip()
        if l1.startswith('1') and l2.startswith('2'):
            try:
                return parse_tle_entry(name, l1, l2)
            except ValueError:
                return None

    return None
