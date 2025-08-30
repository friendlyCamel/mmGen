import numpy as np

# B=139.987*28=3919.636MHz
radar_loc = [0, 0, 0]
light_v = 3e8
f_0 = 60e9
tx_num = 3
rx_num = 4
lamda = light_v / f_0
sample_rate = 10000e3
sample_num = 256
chirp_num = 255                 # 一秒15帧，则一个chirp对应261.44us 所以理论上应该要prep达到261.44us
frequency_slope = 139.987e12
chirp_ramp_time =28e-6
IDLE_TIME = 233e-6
chirp_period = chirp_ramp_time + IDLE_TIME
velocity_res = lamda / chirp_period / tx_num / chirp_num / 2
frame_num = 2
frame_period = 66.666665e-3
angle_padded_num = 64
wavelength = light_v / f_0
azimuth_angle = np.arange(-60, 60, 1)
elevation_angle = np.arange(-30, 30, 1)
azimuth_pattern = 20 * np.log10(np.cos(azimuth_angle / 120 * np.pi) + 0.00001) + 48
elevation_pattern = 20 * np.log10(np.cos(elevation_angle / 60 * np.pi) + 0.00001) + 48




class Radar:
    def __init__(
        self,
        transmitter,
        receiver,
        location=(0, 0, 0),
        speed=(0, 0, 0),
        rotation=(0, 0, 0),
        rotation_rate=(0, 0, 0),
        time=0,
        interf=None,
        seed=None,
        **kwargs
    ):
        self.time_prop = {
            "frame_size": np.size(time),
            "frame_start_time": np.array(time),
        }
        self.sample_prop = {
            "samples_per_pulse": int(
                transmitter.waveform_prop["pulse_length"] * receiver.bb_prop["fs"]
            )
        }
        self.array_prop = {
            "size": (
                transmitter.txchannel_prop["size"] * receiver.rxchannel_prop["size"]
            ),
            "virtual_array": np.repeat(
                transmitter.txchannel_prop["locations"],
                receiver.rxchannel_prop["size"],
                axis=0,
            )
            + np.tile(
                receiver.rxchannel_prop["locations"],
                (transmitter.txchannel_prop["size"], 1),
            ),
        }
        self.radar_prop = {

            "transmitter": transmitter,
            "receiver": receiver,
            "interf": interf,
        }
        # timing properties
        self.time_prop["timestamp"] = self.gen_timestamp()
        self.time_prop["timestamp_shape"] = np.shape(self.time_prop["timestamp"])
        self.process_radar_motion(
            location,
            speed,
            rotation,
            rotation_rate,
        )

    def gen_timestamp(self):
        channel_size = self.array_prop["size"]
        rx_channel_size = self.radar_prop["receiver"].rxchannel_prop["size"]
        pulses = self.radar_prop["transmitter"].waveform_prop["pulses"]
        samples = self.sample_prop["samples_per_pulse"]
        crp = self.radar_prop["transmitter"].waveform_prop["prp"]
        delay = self.radar_prop["transmitter"].txchannel_prop["delay"]
        fs = self.radar_prop["receiver"].bb_prop["fs"]

        chirp_delay = np.tile(
            np.expand_dims(np.expand_dims(np.cumsum(crp) - crp[0], axis=1), axis=0),
            (channel_size, 1, samples),
        )

        tx_idx = np.arange(0, channel_size) / rx_channel_size
        tx_delay = np.tile(
            np.expand_dims(np.expand_dims(delay[tx_idx.astype(int)], axis=1), axis=2),
            (1, pulses, samples),
        )

        timestamp = (
                tx_delay
                + chirp_delay
                + np.tile(
            np.expand_dims(np.expand_dims(np.arange(0, samples), axis=0), axis=0),
            (channel_size, pulses, 1),
        )
                / fs
        )

        if self.time_prop["frame_size"] > 1:
            toffset = np.repeat(
                np.tile(
                    np.expand_dims(
                        np.expand_dims(self.time_prop["frame_start_time"], axis=1),
                        axis=2,
                    ),
                    (
                        1,
                        self.radar_prop["transmitter"].waveform_prop["pulses"],
                        self.sample_prop["samples_per_pulse"],
                    ),
                ),
                channel_size,
                axis=0,
            )

            timestamp = (
                    np.tile(timestamp, (self.time_prop["frame_size"], 1, 1)) + toffset
            )
        elif self.time_prop["frame_size"] == 1:
            timestamp = timestamp + self.time_prop["frame_start_time"]

        return timestamp
    def process_radar_motion(self, location, speed, rotation, rotation_rate):
        shape = self.time_prop["timestamp_shape"]
        if any(
            np.size(var) > 1
            for var in list(location)
            + list(speed)
            + list(rotation)
            + list(rotation_rate)
        ):
            self.radar_prop["location"] = np.zeros(shape + (3,))
            self.radar_prop["speed"] = np.zeros(shape + (3,))
            self.radar_prop["rotation"] = np.zeros(shape + (3,))
            self.radar_prop["rotation_rate"] = np.zeros(shape + (3,))

            for idx in range(0, 3):
                if np.size(speed[idx]) > 1:
                    self.radar_prop["speed"][:, :, :, idx] = speed[idx]
                else:
                    self.radar_prop["speed"][:, :, :, idx] = np.full(shape, speed[idx])

                if np.size(location[idx]) > 1:
                    self.radar_prop["location"][:, :, :, idx] = location[idx]
                else:
                    self.radar_prop["location"][:, :, :, idx] = (
                        location[idx] + speed[idx] * self.time_prop["timestamp"]
                    )

                if np.size(rotation_rate[idx]) > 1:
                    self.radar_prop["rotation_rate"][:, :, :, idx] = np.radians(
                        rotation_rate[idx]
                    )

                else:
                    self.radar_prop["rotation_rate"][:, :, :, idx] = np.full(
                        shape, np.radians(rotation_rate[idx])
                    )

                if np.size(rotation[idx]) > 1:
                    self.radar_prop["rotation"][:, :, :, idx] = np.radians(
                        rotation[idx]
                    )
                else:
                    self.radar_prop["rotation"][:, :, :, idx] = (
                        np.radians(rotation[idx])
                        + np.radians(rotation_rate[idx]) * self.time_prop["timestamp"]
                    )

        else:
            self.radar_prop["speed"] = np.array(speed)
            self.radar_prop["location"] = np.array(location)
            self.radar_prop["rotation"] = np.radians(rotation)
            self.radar_prop["rotation_rate"] = np.radians(rotation_rate)



class Transmitter:
    def __init__(
        self,
        f,
        t,
        tx_power=0,
        pulses=1,
        prp=None,
        channels=None,
    ):
        self.rf_prop = {}
        self.waveform_prop = {}
        self.txchannel_prop = {}
        self.rf_prop["tx_power"] = tx_power

        if isinstance(f, (list, tuple, np.ndarray)):
            f = np.array(f)
        else:
            f = np.array([f, f])

        if isinstance(t, (list, tuple, np.ndarray)):
            t = np.array(t) - t[0]
        else:
            t = np.array([0, t])

        self.waveform_prop["f"] = f
        self.waveform_prop["t"] = t
        self.waveform_prop["bandwidth"] = np.max(f) - np.min(f)
        self.waveform_prop["pulse_length"] = t[-1]
        self.waveform_prop["pulses"] = pulses

        if prp is None:
            prp = self.waveform_prop["pulse_length"] + np.zeros(pulses)
        else:
            if isinstance(prp, (list, tuple, np.ndarray)):
                prp = np.array(prp)
            else:
                prp = prp + np.zeros(pulses)
        self.waveform_prop["prp"] = prp
        self.waveform_prop["pulse_start_time"] = np.cumsum(prp) - prp[0]


        if channels is None:
            channels = [{"location": (0, 0, 0)}]
        self.txchannel_prop = self.process_txchannel_prop(channels)

    def process_txchannel_prop(self, channels):
        # number of transmitter channels
        txch_prop = {}

        txch_prop["size"] = len(channels)

        txch_prop["delay"] = np.zeros(txch_prop["size"])
        txch_prop["grid"] = np.zeros(txch_prop["size"])
        txch_prop["locations"] = np.zeros((txch_prop["size"], 3))
        txch_prop["polarization"] = np.zeros((txch_prop["size"], 3))
        txch_prop["waveform_mod"] = []

        # pulse modulation parameters
        txch_prop["pulse_mod"] = np.ones(
            (txch_prop["size"], self.waveform_prop["pulses"]), dtype=complex
        )

        # azimuth patterns
        txch_prop["az_patterns"] = []
        txch_prop["az_angles"] = []

        # elevation patterns
        txch_prop["el_patterns"] = []
        txch_prop["el_angles"] = []

        # antenna peak gain
        # antenna gain is calculated based on azimuth pattern
        txch_prop["antenna_gains"] = np.zeros((txch_prop["size"]))

        for tx_idx, tx_element in enumerate(channels):
            txch_prop["delay"][tx_idx] = tx_element.get("delay", 0)
            txch_prop["grid"][tx_idx] = tx_element.get("grid", 1)

            txch_prop["locations"][tx_idx, :] = np.array(tx_element.get("location"))
            txch_prop["polarization"][tx_idx, :] = np.array(
                tx_element.get("polarization", [0, 0, 1])
            )

            txch_prop["waveform_mod"].append(
                self.process_waveform_modulation(
                    tx_element.get("mod_t", None),
                    tx_element.get("amp", None),
                    tx_element.get("phs", None),
                )
            )

            txch_prop["pulse_mod"][tx_idx, :] = self.process_pulse_modulation(
                tx_element.get("pulse_amp", np.ones((self.waveform_prop["pulses"]))),
                tx_element.get("pulse_phs", np.zeros((self.waveform_prop["pulses"]))),
            )

            # azimuth pattern
            az_angle = np.array(tx_element.get("azimuth_angle", [-90, 90]))
            az_pattern = np.array(tx_element.get("azimuth_pattern", [0, 0]))
            if len(az_angle) != len(az_pattern):
                raise ValueError(
                    "Lengths of `azimuth_angle` and `azimuth_pattern` \
                        should be the same"
                )

            txch_prop["antenna_gains"][tx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - txch_prop["antenna_gains"][tx_idx]

            txch_prop["az_angles"].append(az_angle)
            txch_prop["az_patterns"].append(az_pattern)

            # elevation pattern
            el_angle = np.array(tx_element.get("elevation_angle", [-90, 90]))
            el_pattern = np.array(tx_element.get("elevation_pattern", [0, 0]))
            if len(el_angle) != len(el_pattern):
                raise ValueError(
                    "Lengths of `elevation_angle` and `elevation_pattern` \
                        should be the same"
                )
            el_pattern = el_pattern - np.max(el_pattern)

            txch_prop["el_angles"].append(el_angle)
            txch_prop["el_patterns"].append(el_pattern)

        return txch_prop
    def process_waveform_modulation(self, mod_t, amp, phs):

        if phs is not None and amp is None:
            amp = np.ones_like(phs)
        elif phs is None and amp is not None:
            phs = np.zeros_like(amp)

        if mod_t is None or amp is None or phs is None:
            return {"enabled": False, "var": None, "t": None}

        if isinstance(amp, (list, tuple, np.ndarray)):
            amp = np.array(amp)
        else:
            amp = np.array([amp, amp])

        if isinstance(phs, (list, tuple, np.ndarray)):
            phs = np.array(phs)
        else:
            phs = np.array([phs, phs])

        if isinstance(mod_t, (list, tuple, np.ndarray)):
            mod_t = np.array(mod_t)
        else:
            mod_t = np.array([0, mod_t])

        if len(amp) != len(phs):
            raise ValueError("Lengths of `amp` and `phs` should be the same")

        mod_var = amp * np.exp(1j * phs / 180 * np.pi)

        if len(mod_t) != len(mod_var):
            raise ValueError("Lengths of `mod_t`, `amp`, and `phs` should be the same")

        return {"enabled": True, "var": mod_var, "t": mod_t}

    def process_pulse_modulation(self, pulse_amp, pulse_phs):
        if len(pulse_amp) != self.waveform_prop["pulses"]:
            raise ValueError("Lengths of `pulse_amp` and `pulses` should be the same")
        if len(pulse_phs) != self.waveform_prop["pulses"]:
            raise ValueError("Length of `pulse_phs` and `pulses` should be the same")

        return pulse_amp * np.exp(1j * (pulse_phs / 180 * np.pi))

class Receiver:
    def __init__(
        self,
        fs,
        noise_figure=10,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=None,
    ):
        self.rf_prop = {}
        self.bb_prop = {}
        self.rxchannel_prop = {}

        self.rf_prop["rf_gain"] = rf_gain
        self.rf_prop["noise_figure"] = noise_figure

        self.bb_prop["fs"] = fs
        self.bb_prop["load_resistor"] = load_resistor
        self.bb_prop["baseband_gain"] = baseband_gain
        self.bb_prop["bb_type"] = bb_type
        if bb_type == "complex":
            self.bb_prop["noise_bandwidth"] = fs
        elif bb_type == "real":
            self.bb_prop["noise_bandwidth"] = fs / 2

        self.validate_bb_prop(self.bb_prop)

        # additional receiver parameters
        if channels is None:
            channels = [{"location": (0, 0, 0)}]

        self.rxchannel_prop = self.process_rxchannel_prop(channels)
    def validate_bb_prop(self, bb_prop):
        if bb_prop["bb_type"] != "complex" and bb_prop["bb_type"] != "real":
            raise ValueError("Invalid baseband type")
    def process_rxchannel_prop(self, channels):
        rxch_prop = {}

        rxch_prop["size"] = len(channels)

        rxch_prop["locations"] = np.zeros((rxch_prop["size"], 3))
        rxch_prop["polarization"] = np.zeros((rxch_prop["size"], 3))

        rxch_prop["az_patterns"] = []
        rxch_prop["az_angles"] = []

        rxch_prop["el_patterns"] = []
        rxch_prop["el_angles"] = []

        rxch_prop["antenna_gains"] = np.zeros((rxch_prop["size"]))

        for rx_idx, rx_element in enumerate(channels):
            rxch_prop["locations"][rx_idx, :] = np.array(rx_element.get("location"))
            rxch_prop["polarization"][rx_idx, :] = np.array(
                rx_element.get("polarization", [0, 0, 1])
            )

            # azimuth pattern
            az_angle = np.array(rx_element.get("azimuth_angle", [-90, 90]))
            az_pattern = np.array(rx_element.get("azimuth_pattern", [0, 0]))
            if len(az_angle) != len(az_pattern):
                raise ValueError(
                    "Lengths of `azimuth_angle` and `azimuth_pattern` \
                        should be the same"
                )

            rxch_prop["antenna_gains"][rx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - rxch_prop["antenna_gains"][rx_idx]

            rxch_prop["az_angles"].append(az_angle)
            rxch_prop["az_patterns"].append(az_pattern)

            # elevation pattern
            el_angle = np.array(rx_element.get("elevation_angle", [-90, 90]))
            el_pattern = np.array(rx_element.get("elevation_pattern", [0, 0]))
            if len(el_angle) != len(el_pattern):
                raise ValueError(
                    "Lengths of `elevation_angle` and `elevation_pattern` \
                        should be the same"
                )
            el_pattern = el_pattern - np.max(el_pattern)

            rxch_prop["el_angles"].append(el_angle)
            rxch_prop["el_patterns"].append(el_pattern)

        return rxch_prop