import numpy as np
from enum import IntEnum

SECONDS = 3600

class BatAction(IntEnum):
    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2


class Battery:
    def __init__(self, dt, efficiency, max_power, capacity, initial_soc=0.5):
        self.dt = dt
        self.efficiency = self._validate_efficiency(efficiency)
        self.max_power = max_power
        self.capacity = capacity
        self.max_energy = self.max_energy_from_power(self.max_power)
        self.set_soc(initial_soc)

    def set_soc(self, soc):
        self._soc = self._validate_soc(soc)

    @property
    def soc(self):
        return self._soc # np.atleast_1d(self._soc)[0]

    def _validate_efficiency(self, efficiency):
        if not (0 < efficiency <= 1):
            raise ValueError("Efficiency must be between 0 and 1.")
        return efficiency

    def _validate_soc(self, soc):
        if not (0 <= soc <= 1):
            raise ValueError("Initial state of charge must be between 0 and 1.")
        return soc

    def max_energy_from_power(self, power):
        return power * (self.dt / SECONDS)

    def _perform_energy_transfer(self, energy, is_charge):
        transferable_energy = min(energy, self.max_energy)
        if is_charge:
            if self._soc >= 1.0:
                return 0.0
            actual_energy = transferable_energy * self.efficiency
            v_soc = self._soc + actual_energy / self.capacity
            if v_soc > 1.0:
                actual_energy = (1.0 - self._soc) * self.capacity
                transferable_energy = actual_energy / self.efficiency
                self._soc = 1.0
            else:
                self._soc = v_soc
            assert self._soc <= 1.0, f"State of charge cannot exceed 1.0: {self._soc}"
            assert self._soc <= v_soc, f"State of charge cannot decrease during charging: {self._soc} > {v_soc}"
        else:
            if self._soc <= 0.0:
                return 0.0
            # devided to get the actual energy from the battery that is need to get the disired energy demanded
            actual_energy = transferable_energy / self.efficiency
            v_soc = self._soc - actual_energy / self.capacity
            if v_soc < 0.0:
                actual_energy = self._soc * self.capacity
                transferable_energy = actual_energy * self.efficiency
                self._soc = 0.0
            else:
                self._soc = v_soc
            assert self._soc >= 0.0, f"State of charge cannot be negative: {self._soc}"
            assert self._soc >= v_soc, f"State of charge cannot increase during discharging: {self._soc} < {v_soc}"
        return transferable_energy if is_charge else -transferable_energy

    def discharge(self):
        load_energy = max(0.0, self._soc * self.capacity)
        return self._perform_energy_transfer(load_energy, is_charge=False)

    def charge(self):
        return self._perform_energy_transfer(self.max_energy, is_charge=True)

    def idle(self):
        return 0.0

    def continuous_action(self, action):
        if action > 1.0 or action < -1.0:
            raise ValueError(f"Action must be between -1.0 and 1.0, but got: {action}")
        if action == 0:
            return self.idle()
        elif action > 0:
            energy = self.max_energy_from_power(action * self.max_power)
            return self._perform_energy_transfer(energy, is_charge=True)
        elif action < 0:
            # removing the min restricition: min(load, self._soc * self.capacity)
            # it should also be possible to sell electricity at certain time periods
            load_energy = self._soc * self.capacity
            energy = self.max_energy_from_power(abs(action) * self.max_power)
            total_discharge_energy = min(load_energy, energy)
            return self._perform_energy_transfer(
                total_discharge_energy, is_charge=False
            )

