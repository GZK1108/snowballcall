import numpy as np
from matplotlib import pyplot as plt


# 该函数使用蒙特卡洛模拟股指未来的价格，使用更精确的方法，G=lnS
def simulation(S, r, T, sigma, N, k_in, steps, plotpath, plothist):
    """
    S: 初始价格
    T: 到期期限（年）
    sigma: 波动率
    N: 路径数
    k_in: 敲入点
    steps: 步长，即计算次数
    """
    delta_t = float(T)/steps
    Spath = np.zeros((steps + 1, N))
    Spath[0] = S

    for t in range(1, steps + 1):
        num = int(N/2)
        z1 = np.random.standard_normal(num)
        z2 = -z1
        z = np.hstack((z1, z2))  # 相当于生成绝对值相等的一组的随机数，保证e^a中a对称
        Spath[t, 0:N] = Spath[t-1, 0:N] * np.exp((r - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z)  # 矩阵运算，提高速度

    up = np.sum(Spath[-1,0:N] > S)
    print("指数上涨的概率为：",up/N)
    if plotpath:
        plt.plot(Spath[:, :])
        plt.plot([k_in]*len(Spath))
        plt.xlabel('time')
        plt.ylabel('price')
        plt.title('Price Simulation')
        plt.grid(True)
        plt.show()
        plt.close()

    if plothist:
        plt.hist(Spath[int(T*steps)], bins=50)
        plt.show()
        plt.close()

    return Spath


def snowball_cashflow(price_path, coupon, N, plothist):
    '''
    :param price_path: 期末股指
    :param coupon: 息票率
    :param N: 模拟次数
    :param plothist:是否作图
    '''
    payoff = np.zeros(N)
    knock_out_times = 0
    knock_in_times = 0
    existence_times = 0
    for i in range(N):
        # 收盘价超过敲出线的交易日
        tmp_up_d = np.where(price_path[:, i] > k_out)
        # 收盘价超出敲出线的观察日(按月观察)
        tmp_up_m = tmp_up_d[0][tmp_up_d[0] % 21 == 0]
        # 收盘价超出敲出线的观察日（超过封闭期）
        tmp_up_m_md = tmp_up_m[tmp_up_m > lock_period]
        tmp_dn_d = np.where(price_path[:, i] < k_in)
        # 根据合约条款判断现金流

        # 情景1：发生过向上敲出
        if len(tmp_up_m_md) > 0:
            t = tmp_up_m_md[0]
            payoff[i] = coupon * (t/252)
            knock_out_times += 1

        # 情景2：未敲出且未敲入
        elif len(tmp_up_m_md) == 0 and len(tmp_dn_d[0]) == 0:
            payoff[i] = coupon
            existence_times += 1

        # 情景3：只发生向下敲入，不发生向上敲出
        elif len(tmp_dn_d[0]) > 0 and len(tmp_up_m_md) == 0:
            # 只有向下敲入，没有向上敲出
            payoff[i] = 0 if price_path[len(price_path)-1][i] > 1 else (price_path[len(price_path)-1][i] - S)
            knock_in_times += 1
        else:
            print(i)


    if plothist:
        plt.hist(payoff, bins=10)
        plt.title('Product Yield Distribution')
        plt.show()
        plt.close()

    return payoff, knock_out_times, knock_in_times, existence_times

# 计算收益的VaR
def var(payoff,price_path,S,c):
    '''
    :param payoff: 产品收益率
    :param price_path: 期末股指
    :param S: 期初股指
    :param c: 分位数
    '''

    temp = np.sort(payoff, axis=0)
    var = np.percentile(temp, c)
    print(f'在置信度为{c}的条件下，产品VaR为：', var)
    temp1 = np.sort(price_path[-1,0:N],axis=0)
    stock_temp = (temp1-S)/S
    var1 = np.percentile(stock_temp, c)
    print(f'在置信度为{c}的条件下，股指收益VaR为：', var1)


if __name__ == "__main__":
    np.random.seed(0)
    sigma = 0.0824
    T = 1
    r = 0.03
    S = 3919.87
    K = S
    N = 30000
    c = 80
    k_in = K * 0.85  # 敲入
    k_out = S * 1.03  # 敲出
    lock_period = 0
    coupon = 0.2
    principal = 1

    steps = 252 * T
    price_path = simulation(S,r, T, sigma, N, k_in=k_in, steps=steps, plotpath=False, plothist=False)

    payoff, knock_out_times, knock_in_times, existence_times = snowball_cashflow(price_path, coupon, N, plothist=False)
    var(payoff,price_path,S,c)
    price = sum(payoff) / len(payoff)
    print('snow_ball price: %f' % price)
    print('knock_out_pro: %f' % (knock_out_times / N))
    print('knock_in_pro: %f' % (knock_in_times / N))
    print('existence_pro: %f' % (existence_times / N))