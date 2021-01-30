
def preprocess_profile(df):
    feature_columns = [
        'ability_score', 'game_times', 'win_game_times', 'in_player_game_times',
        'in_player_win_game_times', 'out_player_win_game_times',
        'role_score_avg', 'role_score_rate_avg', 'wideopen_num_avg',
        'three_point_fieldgoal_rate_avg', 'two_point_fieldgoal_rate_avg',
        'ass_num_avg', 'reb_num_avg', 'blk_num_avg', 'steal_num_avg',
        'floor_num_avg', 'ballpass_num_avg', 'mvp_times_avg',
        'skill_use_times_avg'
    ]
    print(df.shape, df.columns, sep='\n')
    df = df.fillna(0)
    df_tmp = df[feature_columns]
    df = (df_tmp - df_tmp.mean()) / (df_tmp.std())
    # print(df[feature_columns].mean())
    # print(df[feature_columns].std())
    print('-' * 80)
    print(df.shape, df.columns, sep='\n')
    # print(df.head())
    print('-' * 80)
    return df.values
