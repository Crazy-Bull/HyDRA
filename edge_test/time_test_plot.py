import matplotlib.pyplot as plt

# 新数据
time1_new = [0.8635727741888591, 1.1473802264247623, 1.733370859708105, 
             2.3735236057213376, 5.2944252746445795, 8.033052523221288, 
             13.785529275025642, 22.27500558963844, 35.78414734985147]
time2_new = [0.4064974827425821, 0.4297046256916864, 0.458085515669414, 
             0.48617723797048845, 0.5379951958145415, 0.5499501207045147, 
             0.5724941832678659, 0.62918273465974, 0.6433066938604628]
ks = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 绘制第一幅图 (execution time)
plt.figure()
plt.plot(ks, time1_new, marker='o', color='steelblue')
plt.plot(ks, time2_new, marker='^', color='chocolate')
plt.xlabel('k', size=12)
plt.ylabel('average execution time (ms)', size=12)
plt.legend(['ADMM', 'lossless'])
plt.grid(visible=True, which='both')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.savefig('execution_time_plot.png', dpi=600)  # 保存为 600 DPI
plt.close()

print("Plot has been generated and saved as 'execution_time_plot.png' with 600 DPI")