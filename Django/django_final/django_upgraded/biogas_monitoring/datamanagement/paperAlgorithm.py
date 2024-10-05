import random
import matplotlib.pyplot as plt
#Decalre variable of load and biogas
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
#Define Cost Function without Biogas
def costWithoutBio(position):
    position = [[float(element) for element in row] for row in position]
    global exhaustFan
    global wastewaterTreatment
    global coolingPump
    global lighting
    global aeration
    global compensationPump
    global bioGen1
    global bioGen2
    global bioGen3
    global LCOE1
    global LCOE2
    global LCOE3
    global EP
    #Compute Load Demand
    exhaustFanLoad = [a*b for a,b in zip(exhaustFan,position[0])]
    wastewaterTreatmentLoad = [a*b for a,b in zip(wastewaterTreatment, position[1])]
    coolingPumpLoad = [a*b for a,b in zip(coolingPump, position[2])]
    lightingLoad = [a*b for a,b in zip(lighting, position[3])]
    aerationLoad = [a*b for a,b in zip(aeration, position[4])]
    compensationPumpLoad = [a*b for a,b in zip(compensationPump, position[5])]
    loadDemand = [a+b+c+d+e+f for a,b,c,d,e,f in zip(exhaustFanLoad, wastewaterTreatmentLoad, coolingPumpLoad, lightingLoad, aerationLoad, compensationPumpLoad)]
    #Compute Cost
    LWithoutBio = sum([a*b for a,b in zip(loadDemand, EP)])
    return LWithoutBio
#Define Objective Function
def ObjF(position):
    position = [[float(element) for element in row] for row in position]
    global exhaustFan
    global wastewaterTreatment
    global coolingPump
    global lighting
    global aeration
    global compensationPump
    global bioGen1
    global bioGen2
    global bioGen3
    global LCOE1
    global LCOE2
    global LCOE3
    global EP
    #Compute Load Demand and Biogas Generator Capacity
    exhaustFanLoad = [a*b for a,b in zip(exhaustFan,position[0])]
    wastewaterTreatmentLoad = [a*b for a,b in zip(wastewaterTreatment, position[1])]
    coolingPumpLoad = [a*b for a,b in zip(coolingPump, position[2])]
    lightingLoad = [a*b for a,b in zip(lighting, position[3])]
    aerationLoad = [a*b for a,b in zip(aeration, position[4])]
    compensationPumpLoad = [a*b for a,b in zip(compensationPump, position[5])]
    loadDemand = [a+b+c+d+e+f for a,b,c,d,e,f in zip(exhaustFanLoad, wastewaterTreatmentLoad, coolingPumpLoad, lightingLoad, aerationLoad, compensationPumpLoad)]
    bioPowerCapacity1 = [a*b for a,b in zip(position[6], bioGen1)]
    bioPowerCapacity2 = [a*b for a,b in zip(position[7], bioGen2)] 
    bioPowerCapacity3 = [a*b for a,b in zip(position[8], bioGen3)]
    totalBioPowerCapacity = [a+b+c for a,b,c in zip(bioPowerCapacity1, bioPowerCapacity2, bioPowerCapacity3)]
    #Compute Total Energy with using Biogas
    EWithBio = []
    for a, b in zip(loadDemand, totalBioPowerCapacity):
        diff = a - b
        if diff < 0:
            EWithBio.append(0)
        else:
            EWithBio.append(diff)
    #Compute Cost
    LWithBio = sum([a*b for a,b in zip(EWithBio, EP)])+sum(bioPowerCapacity1)*LCOE1+sum(bioPowerCapacity2)*LCOE2+sum(bioPowerCapacity3)*LCOE3
    return LWithBio
#Define Array Value Function
def arrayValue(position):
    position = [[float(element) for element in row] for row in position]
    global exhaustFan
    global wastewaterTreatment
    global coolingPump
    global lighting
    global aeration
    global compensationPump
    global bioGen1
    global bioGen2
    global bioGen3
    global LCOE1
    global LCOE2
    global LCOE3
    global EP
    #Compute Load Demand and Biogas Generator Capacity
    exhaustFanLoad = [a*b for a,b in zip(exhaustFan,position[0])]
    wastewaterTreatmentLoad = [a*b for a,b in zip(wastewaterTreatment, position[1])]
    coolingPumpLoad = [a*b for a,b in zip(coolingPump, position[2])]
    lightingLoad = [a*b for a,b in zip(coolingPump, position[3])]
    aerationLoad = [a*b for a,b in zip(aeration, position[4])]
    compensationPumpLoad = [a*b for a,b in zip(compensationPump, position[5])]
    loadDemand = [a+b+c+d+e+f for a,b,c,d,e,f in zip(exhaustFanLoad, wastewaterTreatmentLoad, coolingPumpLoad, lightingLoad, aerationLoad, compensationPumpLoad)]
    bioPowerCapacity1 = [a*b for a,b in zip(position[6], bioGen1)]
    bioPowerCapacity2 = [a*b for a,b in zip(position[7], bioGen2)] 
    bioPowerCapacity3 = [a*b for a,b in zip(position[8], bioGen3)]
    totalBioPowerCapacity = [a+b+c for a,b,c in zip(bioPowerCapacity1, bioPowerCapacity2, bioPowerCapacity3)]
    #Compute Total Energy with using Biogas
    EWithBio = []
    for a, b in zip(loadDemand, totalBioPowerCapacity):
        diff = a - b
        if diff < 0:
            EWithBio.append(0)
        else:
            EWithBio.append(diff)
    #Compute Cost
    array = [a*b + c*LCOE1 + d*LCOE2 + e*LCOE3 for a,b,c,d,e in zip(EWithBio, EP, bioPowerCapacity1, bioPowerCapacity2, bioPowerCapacity3)]
    return array
#Define Class Particle
class Particle:
    def __init__ (self, position):
        self.position = position
        self.totalValue = ObjF(position)
        self.arrayValue = arrayValue(position)
#Generate random initial position
def randomInitPosition():
    position = np.random.randint(2, size=(9, 24))
    print(position)
    return position
#Check constraint valiable
def checkConstraint(position, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits):
    """
    position: trạng thái của tất cả thiết bị trong 24 giờ (list các list con cho mỗi thiết bị)
    schedules: dict chứa lịch trình và ràng buộc cho từng thiết bị
    max_runtime: thời gian tối đa chạy của các thiết bị
    max_bio_power: công suất tối đa của hệ thống biogas
    max_bio_runtime: thời gian hoạt động tối đa của hệ thống biogas
    biogas_power_limits: tuple chứa các giá trị công suất tối đa cho từng máy phát điện biogas
    """
    position = [[float(element) for element in row] for row in position]
    hours_to_schedule = []
    


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
                #print(position[schedule['index']])
            # Áp dụng công suất nếu có
            '''if schedule['power_mode'] == 'half':
                position[schedule['index']] = [x * 0.5 for x in position[schedule['index']]]'''
        
        # Nếu có yêu cầu lập lịch
        elif schedule['schedule_option'] == 'yes':
            total_hours = schedule['exact_hours']
            
            if schedule['schedule'] == 'day':  # Lập lịch sáng/chiều
                start, end = 6, 18
            elif schedule['schedule'] == 'night':  # Lập lịch đêm
                start, end = 0, 6
                start_2, end_2 = 18, 24  # Đêm có thể là từ 0h-6h và 18h-24h
            elif schedule['schedule'] == 'full_day':  # Cả ngày
                start, end = 0, 24
            
            # Chọn ngẫu nhiên số giờ để bật thiết bị
            if schedule['schedule'] == 'night':  
    # Chọn giờ ngẫu nhiên trong khoảng từ 0h-6h và 18h-24h (đêm)
              hours_to_schedule = random.sample(list(range(start, end)) + list(range(start_2, end_2)), total_hours)

            elif schedule['schedule'] == 'daytime':  
    # Chọn giờ ngẫu nhiên trong khoảng từ 6h-18h (ban ngày)
             hours_to_schedule = random.sample(range(6, 18), total_hours)

            elif schedule['schedule'] == 'full_day':  
    # Chọn giờ ngẫu nhiên trong khoảng từ 0h-24h (cả ngày)
             hours_to_schedule = random.sample(range(0, 24), total_hours)
            
            for hour in hours_to_schedule:
                position[schedule['index']][hour] = 1
            
            # Áp dụng công suất nếu có
            '''if schedule['power_mode'] == 'half':
                position[schedule['index']] = [x * 0.5 for x in position[schedule['index']]]'''

        # Kiểm tra và điều chỉnh nếu tổng thời gian chạy vượt quá max_runtime
            current_runtime = sum(position[schedule['index']])
            if current_runtime > total_hours:
                indices= [i for i, x in enumerate(position[schedule['index']]) if x == 1]

                hours_to_remove = current_runtime - total_hours
                hours_to_turn_off = random.sample(indices, hours_to_remove)
                for hour in hours_to_turn_off:
                    position[schedule['index']][hour] = 0
            else:
                 if current_runtime < total_hours:
                    indices= [i for i, x in enumerate(position[schedule['index']]) if x == 0]
                    hours_to_raise = total_hours - current_runtime
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
            if (0 < (position[6][i]*bioGen1[i]+position[7][i]*bioGen2[i]+position[8][i]*bioGen3[i]) < loadDemand[i]) :
                position[6][i] = 0
                position[7][i] = 0
                position[8][i] = 0
            randomCheck = random.randint(0,1)
            if randomCheck == 0:
               #position[6][i] = 0
               #position[7][i] = 0
               #position[8][i] = 0
            #else:
            #if (position[6][i]+position[7][i]+position[8][i]) >= 1:
               if bioGen1[i] >= loadDemand[i]:
                    position[6][i] = 1
                    position[7][i] = 0
                    position[8][i] = 0
               else:
                   if bioGen2[i] >= loadDemand[i]:
                        position[6][i] = 0
                        position[7][i] = 1
                        position[8][i] = 0
                   else:
                        if  bioGen3[i] >= loadDemand[i]:
                            position[6][i] = 0
                            position[7][i] = 0
                            position[8][i] = 1
                        else:
                            if bioGen1[i]+bioGen2[i] >= loadDemand[i]:
                                position[6][i] = 1
                                position[7][i] = 1
                                position[8][i] = 0
                            else:
                                if bioGen1[i]+bioGen3[i] >= loadDemand[i]:
                                    position[6][i] = 1
                                    position[7][i] = 0
                                    position[8][i] = 1
                                else:
                                    if bioGen2[i]+bioGen3[i] >= loadDemand[i]:
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
        print(totalBioPowerCapacity)
        # Kiểm tra tổng công suất của máy phát
        if float(sum(totalBioPowerCapacity)) > max_bio_power:
            indices = [i for i, x in enumerate(position[6]) if x >= 1]
            if not indices:
                break
            randomIndex = random.choice(indices)
            position[6][randomIndex] = 0
            position[7][randomIndex] = 0
            position[8][randomIndex] = 0
        else:
            break
    
   # Kiểm tra giới hạn thời gian hoạt động của biogas
    while True:
        totalOperatingHour = [a + b + c for a, b, c in zip(position[6], position[7], position[8])]
    
        if float(sum(totalOperatingHour)) > max_bio_runtime:
            indices = [i for i, x in enumerate(totalOperatingHour) if x >= 1]
        
        # Nếu không còn giờ nào để tắt, thoát khỏi vòng lặp
            if not indices:
               break
        
            randomIndex = random.choice(indices)
            position[6][randomIndex] = 0
            position[7][randomIndex] = 0
            position[8][randomIndex] = 0
        else:
            break

    return position

#Algorithm Function
def Algorithm(Pop_Size, MaxT, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits): 
    allPositionBest = None
    allBestValue = float('inf')
    pbest  = None
    bestValue = float('inf')
    particles = []
    #Generate Random Initial Position
    for i in range(Pop_Size):
        initPos = randomInitPosition()
        modifiedPos = checkConstraint(initPos, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits)
        particle = Particle(modifiedPos)
        particles.append(particle)
    #Check best Position
    arrayBestValueOfIterial = []
    for i in range(MaxT):
        #Detect the best position in i_th iterial
        min_object = min(particles, key=lambda obj: obj.totalValue)
        min_index = particles.index(min_object)
        pbest = min_object.position
        bestValue = min_object.totalValue
        bestArrayValue = min_object.arrayValue
        arrayBestValueOfIterial.append(bestValue)
        #Directing other position follow to best position
        for j in range (min_index):
            currentPosition = particles[j].position
            currentArrayValue = particles[j].arrayValue
            for k in range(24):
                if currentArrayValue[k] > bestArrayValue[k]:
                    for l in range(9):
                        currentPosition[l][k] = pbest[l][k]
            newPosition = checkConstraint(currentPosition, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits)
            newParticle = Particle(newPosition)
            particles[j] = newParticle
        for j in range (min_index +1, Pop_Size):
            currentPosition = particles[j].position
            currentArrayValue = particles[j].arrayValue
            for k in range(24):
                if currentArrayValue[k] > bestArrayValue[k]:
                    for l in range(9):
                        currentPosition[l][k] = pbest[l][k]
            newPosition = checkConstraint(currentPosition, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits)
            newParticle = Particle(newPosition)
            particles[j] = newParticle
        print(f"Best position in {i} th Iterial: {pbest} with value {bestValue}")
        print(f"Cost without biogas: {costWithoutBio(pbest)}")
        if bestValue < allBestValue:
            allBestValue = bestValue
            allPositionBest = pbest
    return allPositionBest, allBestValue, arrayBestValueOfIterial
if __name__ == "__main__":
    lastPositionBest = None
    lastBestValue = float('inf')
    popSize = 10
    MaxT = 20
    algorithmIerial = 200
    allValue = []
    max_runtime = 12  # Thời gian tối đa chạy của các thiết bị
    max_bio_power = 500  # Công suất tối đa của hệ thống biogas
    max_bio_runtime = 15  # Thời gian hoạt động tối đa của hệ thống biogas
    biogas_power_limits = (100, 200, 300) 
    schedules = {
    'exhaust_fan': {
        'index': 0,
        'schedule_option': 'no',
        'exact_hours': None,
        'schedule': None,
        'run_time': 'full_day',
        'power_mode': 'normal'
    },
    'wastewater_treatment': {
        'index': 1,
        'schedule_option': 'yes',
        'exact_hours': 12,
        'schedule': 'full_day', 
        'run_time': None,
        'power_mode': 'normal'
    },
    'cooling_pump': {
        'index': 2,
        'schedule_option': 'yes',
        'exact_hours': 6,
        'schedule': 'night',
        'run_time': None,
        'power_mode': 'normal'
    },
    'lighting': {
        'index': 3,
        'schedule_option': 'yes',
        'exact_hours': 6,
        'schedule': 'night',
        'run_time': None,
        'power_mode': 'normal'
    },
    'aeration': {
        'index': 4,
        'schedule_option': 'no',
        'exact_hours': None,
        'schedule': None,
        'run_time': 'full_day',
        'power_mode': 'normal'
    },
    'compensation_pump': {
        'index': 5,
        'schedule_option': 'no',
        'exact_hours': 5,
        'schedule': 'day',
        'run_time': 'full_day',
        'power_mode': 'normal'
    }
}
    for m in range(algorithmIerial):
        PositionBest, BestValue, arrayBestValueOfAlgorithm = Algorithm(popSize, MaxT, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits)
        allValue.append(arrayBestValueOfAlgorithm)
        if BestValue < lastBestValue:
            lastBestValue = BestValue
            lastPositionBest = PositionBest
    print(f"Best position in : {lastPositionBest} with value {lastBestValue}")
    print(f"Cost without biogas: {costWithoutBio(lastPositionBest)}")
    print(f"All value: {allValue}")



