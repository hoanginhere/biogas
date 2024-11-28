def contrast(position, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits):
    """
    position: trạng thái của tất cả thiết bị trong 24 giờ (list các list con cho mỗi thiết bị)
    schedules: dict chứa lịch trình và ràng buộc cho từng thiết bị
    max_runtime: thời gian tối đa chạy của các thiết bị
    max_bio_power: công suất tối đa của hệ thống biogas
    max_bio_runtime: thời gian hoạt động tối đa của hệ thống biogas
    biogas_power_limits: tuple chứa các giá trị công suất tối đa cho từng máy phát điện biogas
    """
    position = [[float(element) for element in row] for row in position]


    # Duyệt qua từng thiết bị để áp dụng lịch trình
    for device, schedule in schedules.items():
        # Khởi tạo trạng thái của thiết bị (24 giờ)
        position[schedule['index']] = [0] * 24
        
        # Nếu không có yêu cầu lập lịch
        if schedule['schedule_option'] == 'no':
            if schedule['run_time'] == 'day':  # Chạy ban ngày
                position[schedule['index']][6:18] = [1] * 12
            elif schedule['run_time'] == 'night':  # Chạy ban đêm
                position[schedule['index']][:6] = [1] * 6
                position[schedule['index']][18:] = [1] * 6
            elif schedule['run_time'] == 'full_day':  # Chạy cả ngày
                position[schedule['index']] = [1] * 24
            # Áp dụng công suất nếu có
            if schedule['power_mode'] == 'half':
                position[schedule['index']] = [x * 0.5 for x in position[schedule['index']]]
        
        # Nếu có yêu cầu lập lịch
        elif schedule['schedule_option'] == 'yes':
            total_hours = schedule['exact_hours']
            
            if schedule['schedule'] == 'morning':  # Lập lịch sáng/chiều
                start, end = 6, 18
            elif schedule['schedule'] == 'night':  # Lập lịch đêm
                start, end = 0, 6
                start_2, end_2 = 18, 24  # Đêm có thể là từ 0h-6h và 18h-24h
            else:  # Cả ngày
                start, end = 0, 24
            
            # Chọn ngẫu nhiên số giờ để bật thiết bị
            if schedule['schedule'] == 'night':
                hours_to_schedule = random.sample(list(range(start, end)) + list(range(start_2, end_2)), total_hours)
            else:
                hours_to_schedule = random.sample(range(start, end), total_hours)
            
            for hour in hours_to_schedule:
                position[schedule['index']][hour] = 1
            
            # Áp dụng công suất nếu có
            if schedule['power_mode'] == 'half':
                position[schedule['index']] = [x * 0.5 for x in position[schedule['index']]]

        # Kiểm tra và điều chỉnh nếu tổng thời gian chạy vượt quá max_runtime
        current_runtime = int(sum(position[schedule['index']]))
        if current_runtime > max_runtime:
            indices= [i for i, x in enumerate(position[schedule['index']]) if x == 1]

            hours_to_remove = current_runtime - max_runtime
            hours_to_turn_off = random.sample(indices, hours_to_remove)
            for hour in hours_to_turn_off:
                position[schedule['index']][hour] = 0
        else:
             if current_runtime < max_runtime:
                indices= [i for i, x in enumerate(position[schedule['index']]) if x == 0]
                hours_to_raise = max_runtime - current_runtime
                hours_to_turn_on = random.sample(indices, hours_to_raise)
                for hour in hours_to_turn_on:
                   position[schedule['index']][hour] = 1



    # 3 Biogas Generation
    # Tải yêu cầu
    exhaustFanLoad = [a * b for a, b in zip(exhaustFan, position[0])]
    wastewaterTreatmentLoad = [a * b for a, b in zip(wastewaterTreatment, position[1])]
    coolingPumpLoad = [a * b for a, b in zip(coolingPump, position[2])]
    lightingLoad = [a * b for a, b in zip(lighting, position[3])]
    aerationLoad = [a * b for a, b in zip(aeration, position[4])]
    compensationPumpLoad = [a * b for a, b in zip(compensationPump, position[5])]
    
    # Tổng tải yêu cầu
    loadDemand = [a + b + c + d + e + f for a, b, c, d, e, f in zip(exhaustFanLoad, wastewaterTreatmentLoad, coolingPumpLoad, lightingLoad, aerationLoad, compensationPumpLoad)]

    # Kiểm tra và điều chỉnh các máy phát điện Biogas
    for i in range(24):
        # Nếu generator không đáp ứng đủ tải, tắt nó
        if (0 < (position[6][i] * bioGen1[i] + position[7][i] * bioGen2[i] + position[8][i] * bioGen3[i]) < loadDemand[i]):
            position[6][i] = 0
            position[7][i] = 0
            position[8][i] = 0
        
        randomCheck = random.randint(0, 1)
        if randomCheck == 0:
            continue  # Nếu randomCheck = 0, bỏ qua bước dưới
        
        # Phân phối generator theo tải yêu cầu
        if bioGen1[i] >= loadDemand[i]:
            position[6][i] = 1
            position[7][i] = 0
            position[8][i] = 0
        elif bioGen2[i] >= loadDemand[i]:
            position[6][i] = 0
            position[7][i] = 1
            position[8][i] = 0
        elif bioGen3[i] >= loadDemand[i]:
            position[6][i] = 0
            position[7][i] = 0
            position[8][i] = 1
        elif bioGen1[i] + bioGen2[i] >= loadDemand[i]:
            position[6][i] = 1
            position[7][i] = 1
            position[8][i] = 0
        elif bioGen1[i] + bioGen3[i] >= loadDemand[i]:
            position[6][i] = 1
            position[7][i] = 0
            position[8][i] = 1
        elif bioGen2[i] + bioGen3[i] >= loadDemand[i]:
            position[6][i] = 0
            position[7][i] = 1
            position[8][i] = 1
        else:
            position[6][i] = 1
            position[7][i] = 1
            position[8][i] = 1

    # Kiểm tra giới hạn công suất tổng của biogas
    while True:
        bioPowerCapacity1 = [a * b for a, b in zip(position[6], bioGen1)]
        bioPowerCapacity2 = [a * b for a, b in zip(position[7], bioGen2)]
        bioPowerCapacity3 = [a * b for a, b in zip(position[8], bioGen3)]
        totalBioPowerCapacity = [a + b + c for a, b, c in zip(bioPowerCapacity1, bioPowerCapacity2, bioPowerCapacity3)]
        
        # Kiểm tra tổng công suất của máy phát
        if sum(totalBioPowerCapacity) > max_bio_power:
            indices = [i for i, x in enumerate(position[6]) if x >= 1]
            randomIndex = random.choice(indices)
            position[6][randomIndex] = 0
            position[7][randomIndex] = 0
            position[8][randomIndex] = 0
        else:
            break
    
    # Kiểm tra giới hạn thời gian hoạt động của biogas
    while True:
        totalOperatingHour = [a + b + c for a, b, c in zip(position[6], position[7], position[8])]
        if sum(totalOperatingHour) > max_bio_runtime:
            indices = [i for i, x in enumerate(totalOperatingHour) if x >= 1]
            randomIndex = random.choice(indices)
            position[6][randomIndex] = 0
            position[7][randomIndex] = 0
            position[8][randomIndex] = 0
        else:
            break

    return position
import random

# Định nghĩa các biến đầu vào
position = [
    [0]*24,  # Exhaust Fan
    [0]*24,  # Wastewater Treatment
    [1]*24,  # Cooling Pump
    [0]*24,  # Lighting
    [0]*24,  # Aeration
    [0]*24,  # Compensation Pump
    [0]*24,  # Biogas Generator 1
    [0]*24,  # Biogas Generator 2
    [0]*24   # Biogas Generator 3
]

schedules = {
    'exhaust_fan': {
        'index': 0,
        'schedule_option': 'no',
        'run_time': 'full_day',
        'power_mode': 'normal'
    },
    'wastewater_treatment': {
        'index': 1,
        'schedule_option': 'yes',
        'schedule': 'morning',
        'exact_hours': 12,
        'power_mode': 'normal'
    },
    'cooling_pump': {
        'index': 2,
        'schedule_option': 'no',
        'run_time': 'day',
        'power_mode': 'normal'
    },
    'lighting': {
        'index': 3,
        'schedule_option': 'no',
        'run_time': 'night',
        'power_mode': 'normal'
    },
    'aeration': {
        'index': 4,
        'schedule_option': 'no',
        'run_time': 'full_day',
        'power_mode': 'normal'
    },
    'compensation_pump': {
        'index': 5,
        'schedule_option': 'no',
        'run_time': 'full_day',
        'power_mode': 'normal'
    }
}

max_runtime = 12  # Thời gian tối đa chạy của các thiết bị
max_bio_power = 500  # Công suất tối đa của hệ thống biogas
max_bio_runtime = 15  # Thời gian hoạt động tối đa của hệ thống biogas
biogas_power_limits = (100, 200, 300)  # Công suất tối đa cho từng máy phát điện biogas

# Các biến toàn cục cần định nghĩa
exhaustFan = [26.2]*6 + [52.4]*12 + [26.2]*6
wastewaterTreatment = [5]*24
coolingPump = [8.8]*6 + [17.6]*12 + [8.8]*6
lighting = [4.4]*24
aeration = [10]*24
compensationPump = [2]*24
bioGen1 = [60]*24
bioGen2 = [80]*24
bioGen3 = [100]*24
LCOE1 = 1000
LCOE2 = 1200
LCOE3 = 1600
EP = [1044]*4 + [1649]*6 + [2973]*2 + [1649]*5 + [2973]*3 + [1649]*2 + [1044]*2

# Gọi hàm contrast
result = contrast(position, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits)

# In kết quả
for i, pos in enumerate(result):
    print(f"Device {i}: {pos}")
