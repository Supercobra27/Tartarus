%% mc_bench.erl
-module(mc_bench).
-export([benchmark/0]).

benchmark() ->
    WorkersList = [1, 2, 4, 8, 16],
    Sims = 100000,
    Trials = 5,
    S0 = 100,
    K = 100,
    R = 0.05,
    Sigma = 0.2,
    T = 1.0,

    {ok, File} = file:open("mc_option_bench.csv", [write]),

    io:format(File, "Workers,Trial,Price,StdErr,CI_low,CI_high,Time_ms~n", []),

    lists:foreach(
      fun(W) ->
          lists:foreach(
            fun(Trial) ->
                Start = erlang:monotonic_time(millisecond),
                {Price, StdErr, CI_low, CI_high} = mc_option:run(W, Sims, S0, K, R, Sigma, T),
                End = erlang:monotonic_time(millisecond),
                TimeMs = End - Start,

                io:format("Workers: ~p, Trial: ~p, Price: ~p, Time(ms): ~p~n",
                          [W, Trial, Price, TimeMs]),

                io:format(File, "~p,~p,~p,~p,~p,~p,~p~n",
                          [W, Trial, Price, StdErr, CI_low, CI_high, TimeMs])
            end,
            lists:seq(1, Trials)
          )
      end,
      WorkersList
    ),

    file:close(File),

    io:format("Benchmark complete. Results written to mc_option_bench.csv~n", []).
