**雪球期权金融衍生品设计**

# **沪深300指数雪球期权产品设计**

---

**【摘要】** 本文设计了一种挂钩沪深300指数的雪球期权产品。本产品收益率位于0%-20%之间，但收益为0%的概率较小。在理财产品持有期间，当标的合约收盘价高于敲出价格时，产品终止，投资者可以获得20%*(t/T)的收益；当标的合约在产品期限内收盘价从未高于敲出价格，且从未低于敲入价格，投资者可获得最高收益率20%；若标的资产到期日价格低于敲入价格，投资者的收益为0。VaR分析显示，本产品风险较小，能实现保本，同时有95%的把握投资者收益大于1.67%，且有25%的把握投资者可以取得20%的高收益率，对相信标的资产在未来一段时间内价格波动稳定、不会大起大落的投资者具有较大的吸引力，并且本产品可以根据客户需要，定制投资时长，能满足不同偏好的投资者需要。

## 一、产品介绍

本产品为一款挂钩沪深300指数的雪球期权金融产品。其具有自动赎回的特性，雪球结构是在提供一定程度下跌保护的同时表达温和看涨的观点，只要挂钩标的资产价格不发生大跌，持有期限越长获利就越高。本产品在预先设定一个标的资产的价格区间，区间的最小值为敲入价格，最大值为敲出价格，标的资产的期初价格位于这个区间之间。如果标的指数在合约期限内某个交易日的收盘价低于敲入水平时，其收益为0；如果标的指数在合约期限内的观察日的收盘高于敲出水平时，则提前终止合约并获得计息天数下的年化票息收益；如果标的指数先敲入之后再敲出，收益按敲出条款；如果标的指数在合约期限内既未敲出也未敲入时，则获得全部年化票息收益。

雪球期权通过设置价格区间，投资者的收益随挂钩的标的涨跌情况变化，在一定时间下限制了投资者的最低收益率和最高收益率；当投资者投资时间越长，其获得的收益的期望值也就越大，对于有长期投资需求的投资者友好。并且我们提供保底，即使没有收益，也不会带来本金的损失。该产品对于预期沪深300指数在未来一段时间内价格波动较小的投资者具有较大的吸引力。

## 二、产品说明书

1. 本产品说明书

| 产品名称 | 沪深300雪球期权理财产品 |
| --- | --- |
| 币种 | 人民币 |
| 挂钩标的 | 沪深300指数（000300.SH） |
| 认购起点 | 1元人民币为1份，认购起点份额为5万份，超过存款起点的金额部分，应为1千元人民币的整数倍 |
| 申购/赎回 | 本理财产品存续期内不提供主动申购和赎回，可能因为标的资产价格变动自动提前终止。 |
| 认购期 | 2022年5月1日10:00:00至2022年5月10日11:00:00 |
| 成立日 | 2022年5月10日（遇非工作日则顺延至下一工作日），理财产品自成立日起计算收益 |
| 到期日 | 2023年5月10日（遇非工作日则顺延至下一工作日） |
| 理财产品期限 | 365天，自本理财产品成立日（含）至本理财产品到期日（不含） |
| 定盘价格 | 中证指数有限公司公布的沪深300指数收盘价，沪深300指数的证券代码为“000300.SH” |
| 期初价格 | 成立日当日的定盘价格 |
| 期末价格 | 结算日当日的定盘价格 |
| 敲入边界 | 期初价格的85.00%（按照舍位法精确到小数点后2为） |
| 敲出边界 | 期初价格的103.00%（按照舍位法精确到小数点后2为） |
| 敲入观察日 | 每个交易日（252个） |
| 敲出观察日 | 每个月最后一个交易日（12个） |
| 最低收益率 | 0% |
| 息票率 | 20.0% |
| 参与率 | 100.00% |
| 本金及理财收益支付 | 1.      若观察日未发生敲出，本理财产品到期日或提前终止日一次性支付本金及理财收益（如有）。
2.      若观察日发生敲出，则敲出第二天一次性支付本金及收益。详情见“本金及理财收益支付” |
| 费用 | 1、  销售费：本理财产品收取销售费率0.30%/年
2、  托管费：本理财产品收取托管费率0.02%/年
3、  固定投资管理费：本理财产品收取固定投资管理费率0.20%/年 |
| 收益计算单位 | 每1万份为1个收益计算单位，每收益计算单位理财收益按照四舍五入法精确到小数点后2位 |
| 清算期 | 认购登记日到成立日期间为认购清算期，到期日或提前终止日到理财资金返还到账日为还本清算期，认购清算期和还本清算期内不计付利息 |
1. 本金及理财收益
    
    理财浮动收益率与沪深300指数水平挂钩。本理财产品所指沪深300指数为沪深300指数市场交易价格。理财收益率为扣除理财产品相关费率后的理财收益率。
    
2. 收益率的确定
    
    理财收益率根据以下约定来确定（理财收益率未扣除衍生交易相关税费）：
    （1）若沪深300指数定盘价格在观察期内始终位于敲入与敲出边界以内，则获得收益率为20%（年化），在此情况下，每收益计算单位理财收益的计算公式为：
    每收益计算单位理财收益＝收益计算单位份额×理财收益率×理财产品期限÷365
    （每收益计算单位理财收益按照四舍五入法精确到小数点后2位）
    （2）若沪深300指数定盘价格在观察期内高于或等于敲出边界，则理财收益率为：
    理财收益率（年化）＝20%*(t/T)
    其中，t为期初到观察到指数超出敲出边界的时长，若沪深300指数定盘价格在观察期内先小于敲入边界后大于敲出边界，与上述情况相同。
    理财收益率按照舍位法精确到小数点后2位
    在此情况下，每收益计算单位理财收益的计算公式为：
    每收益计算单位理财收益＝收益计算单位份额×理财收益率×理财产品期限÷365
    （每收益计算单位理财收益按照四舍五入法精确到小数点后2位）
    （3）若沪深300指数定盘价格在观察期内低于或等于敲入边界，且沪深300指数期末价格仍然低于敲入边界，则投资收益率为0（年化）：
    每收益计算单位理财收益＝收益计算单位份额×理财收益率×理财产品期限÷365
    （每收益计算单位理财收益按照四舍五入法精确到小数点后2位）
    

## 三、产品收益分析

本理财产品是挂钩沪深300指数的雪球期权产品，敲入价格为成立日定盘价格的85%，敲出价格为定盘价格的103%，敲出后的收益率为20%*(t/T)，最低收益率为0，能实现保本，最高收益率约为20%，收益较高。该理财产品在正常情况下不会导致投资者本金亏损（不考虑相关手续费），并且根据观察日沪深300指数与期初指数的大小来判断投资者能获得多少收益。相当于投资者卖空一个具有障碍价格的看跌期权，同时使用组合保险进行保本。
产品到期收益如下，其中S表示成立日沪深300股指的价格：

![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled.png)

具体收益如表2所示：

|  | 标的资产到期价格 | 收益率 |
| --- | --- | --- |
| 标的资产在合约期间不超过敲入价格与敲出价格 |  | 20% |
| 标的资产在合约期间超过了敲出价格 | S(t)，提前终止 | 20%*(t/T) |
| 标的资产在合约期间到期时小于敲入价格 |  | 0 |

选取沪深300指数在2021年5月10日至2022年5月10日期间一年的收益率计算波动率，得到标的资产的年化波动率为0.0824，假设收益率为0.03。标的期初价格S_0=3919.87，K\_in=0.85S_0=3331.89,K\_out=1.03S_0=4037.47。使用蒙特卡罗模拟，假设沪深300指数收益率符合对数正态分布，用30000条轨道来模拟产品发行后365天内沪深300指数的价格走势，接下来将展示蒙特卡罗模拟得到的结果。

## 四、模拟分析

1. 模拟方法
    
    蒙特卡罗是一种通过模拟标的资产价格随机运动路径得到衍生品价格的数值方法。蒙特卡罗的优点在于可以对各种复杂衍生产品进行定价，思想较为直接，但是在用此方法对本文标的资产价格进行路径模拟时，对于任意一条路径，我们都要确定在此情况下指数价格是否超出敲入界限或敲出界限，故需要比较标的资产价格每日价格与敲入界限的大小以及标的资产每月最后一个交易日价格与敲出界限的大小。在进行蒙特卡罗模拟的时候，保存每天的标的资产的价格，将其与敲出与敲入界限比较，计算收益率。
    在这里我们对沪深300指数的价格运动趋势进行模拟，得到在产品的收盘价格后，再将收盘价与障碍价格进行比较。本文假设沪深300价格服从几何布朗运动，即：
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%201.png)
    
    将上式离散化，可得：
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%202.png)
    
    其中 $\varepsilon$ 为随机数，服从标准正态分布。 $\hat{\mu}=r_f。\Delta t$ 为一天。
    由于T比较大，为了模拟更加准确，采用更精确的离散化，即组合价格为对数正态分布。
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%203.png)
    
    蒙特卡洛模拟得到沪深300价格的一条路径各个节点的式子如下：
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%204.png)
    
    本文采用Python进行模拟仿真，操作系统为windows 10，编译环境为python 3.9。期初价格（2022年5月10日沪深300收盘价）为3919.87，收益率 为0.03，波动率 为0.0824。本产品时间期限为365天，共252个交易日。
    
2. 模拟路径
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%205.png)
    
    本次共模拟了30000条路径。图2展示了从产品成立日到产品到期日沪深300指数价格的分布路径，共计30000条价格路径。
    
3. 模拟结果分布
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%206.png)
    
    图3展示了产品到期日沪深300收盘价的分布图。产品到期日时点，沪深300指数上涨的概率约为62.78%，下跌的概率约为37.22%。沪深300收盘价绝大部分位于3200点至4900点之间。
    
4. 本产品的收益率区间及分布
    
    本产品收益率区间为0%-20%。产品收益率分布如图4.
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%207.png)
    
    从图4可以看出，随着沪深300股指价格变动，本产品的收益率位于0%-20%。并且本产品实现了保本，且有较大概率获得最大收益率，有投资价值。
    
5. VaR风险分析
    
    VaR指在险价值，表示在给定时间和置信区间下，某项投资或资产组合可能出现的损失的最大值，首先给出如下定义：
    $W_0$表示该资产组合在期初的初始价值；
    $\widetilde{W}$表示该资产组合在期末的预期收益；
    $\hat{r}$表示最低收益率；
    
    $\mu$表示期望收益率；
    $1-\alpha$为给定的置信区间；
    $VaR_{1-\alpha}$表示在该置信区间内损失的最大值，最大损失为投资组合在T时刻期望价值与期末期望价值之差（相对VaR）：
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%208.png)
    
    资产组合损失超过 $VaR_{1-\alpha}$的概率仅为 $\alpha$ ，可以得到
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%209.png)
    
    VaR的计算依赖于资产收益率的分布形式，假设资产收益率服从正态分布的情况下，有：
    
    ![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%2010.png)
    
    其中，  $\sigma$ 为资产组合分布的标准偏差。
    根据蒙特卡罗模拟得到的收益率分布图，计算本产品在不同置信区间下的VaR。（如下所示）
    
    | 置信区间 | 95% | 75% | 50% | 25% | 20% | 15% | 10% | 5% | 1% |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 产品收益率VaR | 1.67% | 3.33% | 8.33% | 20% | 20% | 20% | 20% | 20% | 20% |
    
    VaR分析显示，产品总体收益均为正，在25%的置信区间获得最高收益率20%，并且在99%的置信区间收益仍然为正，收益为0的概率较小，风险较小。
    

## 五、总结

本文设计了一种挂钩沪深300指数的雪球期权产品。本产品收益率由最低收益率、最高收益率和中间收益率组成。在理财产品持有期间，当标的合约收盘价高于敲出价格时，产品终止，投资者可以获得20%*(t/T)的收益；当标的合约在产品期限内收盘价从未高于敲出价格，且从未低于敲入价格，投资者可获得最高收益率20%；若标的资产到期日价格低于敲入价格，投资者的收益为0。VaR分析显示，本产品风险较小，本金有保障，有95%的把握投资者收益大于1.67%，且有25%的把握投资者可以取得20%的高收益率，对相信标的资产在未来一段时间内价格波动稳定，不会大起大落的投资者具有较大的吸引力。

**附录：模拟程序结构图**

![Untitled](%E9%9B%AA%E7%90%83%E6%9C%9F%E6%9D%83%E8%A1%8D%E7%94%9F%E5%93%81%E8%AE%BE%E8%AE%A1%20467cc66e813c4d52a933f8a5a324de61/Untitled%2011.png)
