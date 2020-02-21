#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

# rule 2-1-0: verbs/nouns/others

birth = '(出生|诞生)'
birth_all = f'({birth}|生)'
death = f'(死亡|死掉|过世|逝世|去世|离世|离开人间)'
death_sim = f'死'
death_all = f'({death}|{death_sim})'
_in = '(在|于)'


# rule 2-1-1: Place/Location

place = '(地区|地方|地点|省份|省|城市)'
place_sim = '(地|省)'
place_all = f'({place}|{place_sim})'
country = '(国家|国)'

# rule 2-1-2: what/which

which = '哪(一个|个)?'
which_sim = '(哪|哪一)'
which_all = f'({which}|{which_sim})'
which_place_sim = f'{which_sim}{place_sim}'
whichplace = f'{which}{place}'
whichplace_all = f'({whichplace}|{which_place_sim})'
what = '(甚么|什么)'
what_which = f'({what}|{which_all})'
what_sim = '(啥|何|哪)'

# rule 2-1-3: wh-word for place (where, what place)

whatplace = f'{what}|{place}'
where = '哪(里)'
where_all = f'({whichplace_all}|{whatplace}|{where})'

# rule 2-1-4: date/time

date = '(日期|日子|天)'
date_sim = '日'
month = '月份'
month_sim = '月'
year = '年'
time = '(时后|时候|时间)'
time_sim = '时'
start = '(开工|建立|成立|签署|生效|开始|始于)'

# rule 2-1-5: wh-word for time/date (what time, when)

when = f'({what_which}({time}|{year}|{month}|{date})|{what_which}({time_sim}|{year}|{month_sim}|{date_sim}))'
human = '(人|人物)'
who = f'((({what_which}|{what_sim}).{0, 5}{human})|谁)'
found = f'(创立|创办|开始|创建|发起|创业|建立)'

name = '(?P<name>[^的]{2,10}?)'
is_ = '(叫|是)'
height = '((总)高度|身高)'
have = '(有|拥有)'
strict_label_map = {
    '名字': [f'^{name}{is_}{what}名字', f'^{name}{is_}名字{what}', f'^{name}(的)?名字{is_}{what}'],
    # rule 2-1-a: 地方
    '出生地': [f'^{name}的老家', f'^{name}(的)?{birth}.*{place_all}',
            f'^{name}{birth_all}.*{where_all}', f'{where_all}{is_}{name}的{birth_all}',
            f'^{name}{is_}{where_all}(的)?人', f'^{name}({is_})?{where_all}来'],
    '死亡地': [f'^{name}的死亡地', f'^{name}{death_all}{_in}{place_all}',
            f'^{name}{_in}{where_all}{death_all}',
            f'^{name}{death_all}{_in}{where_all}', f'{where_all}{is_}{name}的{death_all}'],
    '墓地': [],
    # rule 2-1-b: 时间
    '出生日期': [f'^{name}的生日', f'^{name}的出生日期', f'^{name}的出生日',
             f'^{name}{when}.*{birth_all}', f'^{name}的{birth_all}.*{when}'],
    '死亡日期': [f'^{name}的死亡地',
             f'^{name}{when}.*{death_all}', f'^{name}的{death_all}.*{when}'],
    '成立或建立時間': [f'^{name}的{start}.*{time}',
                f'^{name}{when}.*{start}', f'^{name}{start}.*{when}'],
    # rule 2-1-c: 人物资讯
    '國籍': [f'^{name}的国籍', f'^{name}{is_}{what_which}(朝代|国籍)',
           f'^{name}(的)?{birth_all}.*{what_which}({country}|朝代|国籍)', f'^{name}({is_})?{what_which}({country}|朝代|国籍){birth_all}',
           f'^{name}{is_}{what_which}({country}|朝代|国籍)(的)?人', f'^{name}({is_})?{what_which}({country}|朝代)来'],
    '职业': ['职(位|业|务)', '工作', '职称', '岗位'],
    '創辦者': [f'{found}(人|者)', f'{who}.*{found}', f'{start}.*{found}'],
    '高度': [f'^{name}(的)?{height}'],
    '配偶': [f'^{name}(的)?(妻子|老婆|配偶){is_}{what}名字'],
    '父親': [f'^{name}(的)?(爸爸|父亲|老爸){is_}{what}名字'],
    '母親': [f'^{name}(的)?(妈妈|母亲|老母){is_}{what}名字'],
    '子女数目': [f'^{name}总共{have}几个(小孩|孩子)']

}

# not used
alias_map = {
    '配偶': ['妻子', '老婆', '配偶'],
    '父親': ['爸爸', '父亲', '老爸'],
    '母親': ['妈妈','母亲','老妈'],
    '兄弟姊妹': ['兄弟姐妹','兄弟姊妹']
}