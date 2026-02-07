-module(mc_pi).
-export([run/2, worker/3]).

%% run(Workers, SamplesPerWorker)
%% scheduler
run(Workers, Samples) ->
    Parent = self(),

    %% Spawn Workers
    [spawn(?MODULE, worker, [Parent, Samples, Id])
    || Id <- lists:seq(1, Workers)],

    gather(Workers, 0, Workers * Samples).

%% Worker simulates N points
%% kernel
worker(Parent, Samples, _Id) ->
    rand:seed(exsplus, {erlang:monotonic_time(), erlang:phash2(self()), 123}),
    Hits = simulate(Samples, 0), %% Embarrassingly Parallel
    Parent ! {result, Hits}.

%% compute loop
simulate(0, Acc) ->
    Acc;
simulate(N, Acc) ->
    X = rand:uniform(),
    Y = rand:uniform(),

    NewAcc = 
        case X*X + Y*Y =< 1 of
            true -> Acc + 1;
            false -> Acc
        end,
    simulate(N-1, NewAcc).

%% collect results
%% reducer
gather(0, Hits, TotalSamples) ->
    Pi = 4 * Hits / TotalSamples,
    io:format("π ≈ ~p~n", [Pi]);

gather(Remaining, Hits, TotalSamples) ->
    receive
        {result, Count} ->
            gather(Remaining-1, Hits + Count, TotalSamples)
    end.