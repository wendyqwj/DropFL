import matplotlib.pyplot as plt
import os, sys
for dirPath, dirNamesList, logList in os.walk("/home/pity/federated-learning/log"):
    for logName in logList:
        try:
            f = open(dirPath + '/' + logName,'r')
            plt.figure()
            lineList = f.readlines()
            fedList = []
            epo = (int(logName.split('epo')[1].split('_',1)[0]) // 5 * 5) - 1
            for i in range(len(lineList)):
                if lineList[i] == "===================DATA Round 0-{}====================\n".format(epo):
                    
                    for accStr in lineList[i+2:i+6]:
                        accStr = accStr.strip(',\n').strip('[').strip(']')
                        accList = list(map(lambda x: 0.01*x, list(map(float, accStr.split(", ")))))
                        fedList.append(accList)
            
            if epo < 40:
                gap = 1
            elif epo < 500:
                gap = 5
            else:
                gap = 50
            if "case1" in logName:
                plt.ylim(0.85,0.97)
            # if "caseX" in logName:
            #     plt.ylim(0.3,0.6)
            ave = []
            for i in range(len(fedList)):
                ave_i = []
                for j in range(len(fedList[i]) // gap):
                    spi = fedList[i][j*gap:(j+1)*gap]
                    ave_i.append(sum(spi) / len(spi))
                ave.append(ave_i)
            for i in range(len(fedList)):
                if i == 0:
                    plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="no att")
                if i == 1:
                    plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="complete att")
                if i == 2:
                    plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="random att")
                if i == 3:
                    plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="shapley att")
                # if i == 4:
                #     plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="random att 0.5")
                # if i == 5:
                #     plt.plot(list(range(len(fedList[i])))[0::gap], ave[i], label="shapley att 0.5")

            plt.xlabel('Communication Rounds')
            plt.ylabel('Test Accuracy')
            plt.legend()
            plt.savefig('/home/pity/federated-learning/save/{}.jpg'.format(logName))
            print("^Draw ",logName)
        except Exception as e:
            print("## Error!!!", logName, e)
        finally:
            f.close()