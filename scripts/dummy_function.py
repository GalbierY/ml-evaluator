import numpy as np

class Simulation_Env_maneuver:
    def black_box_energy_management(desired_power, regeneration_level):
        """ 
        Black Box simulating the power management of an electric vehicle. 

        Inputs: 
        - desired_power: Float (kW), between 0 and 150 
        - regeneration_level: float, between 0 (without regeneration) and 1 (maximum regeneration) 

        Outputs: 
        - Performance: Float, between 0 and 10 (acceleration response) 
        - Efficiency: Float, between 0 and 10 (regeneration efficiency) 
        """

        print('Run Simulation...')
        #time.sleep(1.0)  # reduced delay for testing

        # === Internal states ===
        base_temp_motor = 40
        motor_temp_gain = 0.6  # mais agressivo
        motor_temp = base_temp_motor + motor_temp_gain * desired_power
        motor_temp = min(motor_temp, 120)

        base_temp_battery = 25
        battery_temp_gain = 0.15  # mais agressivo
        battery_temp = base_temp_battery + battery_temp_gain * desired_power
        battery_temp = min(battery_temp, 60)

        soc = 0.7

        # === Performance ===
        # Non-linear "sweet spot" — mais estreito
        optimal_power_ratio = 0.5
        power_ratio = desired_power / 150
        perf_efficiency_shape = np.exp(-12 * (power_ratio - optimal_power_ratio)**2)  # bell mais estreita

        # Dynamic losses — mais fortes
        temp_factor = 1 + 0.006 * max(0, motor_temp - 60)
        dynamic_losses = (0.1 + 0.35 * power_ratio**2) * temp_factor

        # Inertia effect — maior penalização
        inertia_vehicle = 1.0 + 1.2 * regeneration_level

        # Cross-effect: regen penaliza performance com parabola
        regen_penalty_on_performance = 1.0 - 0.4 * regeneration_level**2

        # Final performance — escala para 0 a 10
        performance = perf_efficiency_shape * (1 - dynamic_losses) / inertia_vehicle * regen_penalty_on_performance
        performance = np.clip(performance, 0, 1)
        performance_scaled = performance * 10

        # === Efficiency ===
        # Cross-effect: penaliza efficiency com power_ratio² (gera parabola)
        power_eff_penalty = np.exp(-6 * power_ratio**2)

        # Battery temp factor — igual
        if battery_temp < 15:
            temp_eff_factor = (battery_temp + 10) / 25
        elif battery_temp > 35:
            temp_eff_factor = max(0.0, 1 - (battery_temp - 35) / 25)
        else:
            temp_eff_factor = 1.0

        # SOC effect — igual
        soc_eff_factor = max(0.0, 1 - soc**3)

        # Regen term com parabola
        regen_term = regeneration_level * (1 - 0.3 * regeneration_level**2)

        # Final efficiency — escala para 0 a 10
        efficiency = regen_term * temp_eff_factor * soc_eff_factor * power_eff_penalty
        efficiency = np.clip(efficiency, 0, 1)
        efficiency_scaled = efficiency * 10

        return -performance_scaled, -efficiency_scaled