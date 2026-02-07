%% mc_option.erl
-module(mc_option).
-export([run/7, worker/8]).

%% Params = #{s0=>100, k=>100, r=>0.05, sigma=>0.2, t=>1.0}

run(Workers, Sims, S0, K, R, Sigma, T) ->
    Parent = self(),

    [spawn_link(?MODULE, worker, [Parent, Sims, S0, K, R, Sigma, T, Id])
    || Id <- lists:seq(1, Workers)],

     %% Double the amount because of antithetic sampling
    gather(Workers, 0.0, 0.0, Workers * Sims * 2, R, T).

worker(Parent, Sims, S0, K, R, Sigma, T, _Id) ->
    rand:seed(exsplus, {erlang:monotonic_time(), erlang:phash2(self()), 123}),

    %% Precompute Constants
    Drift = (R - 0.5 * Sigma * Sigma) * T,
    VolTerm = Sigma * math:sqrt(T),

    ExpDrift = math:exp(Drift),

    {Sum, SumSq} = simulate(Sims, S0, K, ExpDrift, VolTerm, {0.0, 0.0}),

    Parent ! {result, Sum, SumSq}.

simulate(0, _, _, _, _, {Sum, SumSq}) ->
    {Sum, SumSq};
%% Drift = Expected Growth, Diffusion = Randomness
simulate(N, S0, K, ExpDrift, VolTerm, {Sum, SumSq}) ->
    Z = rand:normal(),

    ST1 = S0 * ExpDrift * math:exp(VolTerm * Z),
    ST2 = S0 * ExpDrift * math:exp(-VolTerm * Z),

    %% Arithmetic Sampling
    Payoff1 = max(ST1 - K, 0.0),
    Payoff2 = max(ST2 - K, 0.0),

    NewSum   = Sum + Payoff1 + Payoff2,
    NewSumSq = SumSq + Payoff1*Payoff1 + Payoff2*Payoff2,

    
    simulate(N-1, S0, K, ExpDrift, VolTerm, {NewSum, NewSumSq}).

gather(0, Sum, SumSq, TotalSims, R, T) ->

    Mean = Sum / TotalSims,
    Var = (SumSq - TotalSims * Mean * Mean) / (TotalSims - 1),
    StdErr = math:sqrt(Var/TotalSims),

    CI_low  = Mean - 1.96*StdErr,
    CI_high = Mean + 1.96*StdErr,

    Price = math:exp(-R*T) * Mean,

    io:format("Option price = ~p~n", [Price]),
    io:format("StdErr = ~p~n", [StdErr]),
    io:format("95% CI = [~p, ~p]~n", [CI_low, CI_high]),

    {Price, StdErr, CI_low, CI_high};

gather(Remaining, Sum, SumSq, TotalSims, R, T) ->
    receive
        {result, SumPart, SqPart} ->
            gather(Remaining-1, Sum + SumPart, SumSq + SqPart, TotalSims, R, T)
    end.