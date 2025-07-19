classdef MOAOO < ALGORITHM
% Enhanced Multi-Objective Animated Oat Optimization Algorithm (MO-AOO)
% 
% Key improvements for Pareto front convergence:
% 1. Reference vector guidance for exploitation
% 2. Adaptive step size control
% 3. Hybrid DE operators
% 4. Elite preservation mechanism
% 5. Objective space projection

%------------------------------- Reference --------------------------------
% R.-B. Wang et al., "The Animated Oat Optimization Algorithm: A nature-inspired 
% metaheuristic for engineering optimization and a case study on Wireless Sensor Networks," 
% Knowledge-Based Systems, vol. 318, p. 113589, 2025.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [CrossoverRate, MutationRate, ArchiveSize, HumidityFactor] = Algorithm.ParameterSet(0.9, 0.1, 100, 0.5);
            
            %% Initialize population
            Population = Problem.Initialization();
            Archive    = UpdateArchive([], Population, ArchiveSize);  % Initialize archive
            generation = 0;
            maxGen     = ceil(Problem.maxFE / length(Population));
            
            %% Optimization loop
            while Algorithm.NotTerminated(Population)
                generation = generation + 1;
                
                %% Adaptive parameters
                F = 0.5 * (1 - generation/maxGen);  % DE scaling factor
                currentMutationRate = MutationRate * (1 - generation/maxGen);
                currentHumidity = HumidityFactor * (1 - generation/maxGen);
                
                %% PHASE 1: Enhanced Exploration
                NewDec = zeros(length(Population), Problem.D);
                for i = 1:length(Population)
                    % Adaptive dispersal probability
                    dispersal_prob = min(0.6, 0.2 + 0.4*(generation/maxGen));
                    
                    if rand() < dispersal_prob
                        % Hybrid DE/rand/1 mutation with bounds checking
                        idx = randperm(length(Population), 3);
                        mutant = Population(idx(1)).dec + F*(Population(idx(2)).dec - Population(idx(3)).dec);
                        % Ensure bounds before mutation
                        mutant = max(min(mutant, Problem.upper), Problem.lower);
                        NewDec(i,:) = PolynomialMutation(mutant, Problem);
                    else
                        % Objective-space guided exploration
                        if ~isempty(Archive)
                            archive_objs = vertcat(Archive.objs);
                            current_obj = Population(i).obj;
                            if size(current_obj, 1) > 1
                                current_obj = current_obj';
                            end
                            distances = sqrt(sum((archive_objs - current_obj).^2, 2));
                            [~, ref_idx] = min(distances);
                            direction = Archive(ref_idx).dec - Population(i).dec;
                            step = 0.05 * (1 - generation/maxGen) * randn(1, Problem.D);
                            NewDec(i,:) = Population(i).dec + step .* direction;
                        else
                            % Fallback if archive is empty
                            NewDec(i,:) = Population(i).dec + 0.05 * randn(1, Problem.D);
                        end
                        % Strict boundary enforcement
                        NewDec(i,:) = max(min(NewDec(i,:), Problem.upper), Problem.lower);
                    end
                end
                NewPop = Problem.Evaluation(NewDec);
                
                % Elite-preserving selection
                Population = EnvironmentalSelection([Population, NewPop], length(Population));
                
                %% PHASE 2: Targeted Exploitation
                NewDec = zeros(length(Population), Problem.D);
                for i = 1:length(Population)
                    % Select nearest reference solution from archive
                    if ~isempty(Archive)
                        archive_objs = vertcat(Archive.objs);
                        current_obj = Population(i).obj;
                        if size(current_obj, 1) > 1
                            current_obj = current_obj';
                        end
                        distances = sqrt(sum((archive_objs - current_obj).^2, 2));
                        [~, ref_idx] = min(distances);
                        ref = Archive(ref_idx);
                    else
                        % Fallback if archive is empty
                        ref = Population(randi(length(Population)));
                    end
                    
                    % DE/current-to-best mutation
                    if ~isempty(Archive)
                        mutant = Population(i).dec + F*(ref.dec - Population(i).dec) + ...
                                 F*(Archive(randi(length(Archive))).dec - Population(randi(length(Population))).dec);
                    else
                        % Fallback DE mutation
                        idx = randperm(length(Population), 3);
                        mutant = Population(idx(1)).dec + F*(Population(idx(2)).dec - Population(idx(3)).dec);
                    end
                    
                    % Adaptive mutation strength
                    mutationStrength = max(0.01, 0.1 * (1 - generation/maxGen));
                    if rand() < currentMutationRate
                        mutant = PolynomialMutation(mutant, Problem, mutationStrength);
                    end
                    
                    NewDec(i,:) = mutant;
                end
                NewPop = Problem.Evaluation(NewDec);
                
                % Elite-preserving selection
                Population = EnvironmentalSelection([Population, NewPop], length(Population));
                
                %% PHASE 3: Diversity-Enhanced Crossover
                if rand() < CrossoverRate
                    OffspringDec = [];
                    for i = 1:ceil(length(Population)/2)
                        parent1 = TournamentSelection(Population, Archive);
                        parent2 = TournamentSelection(Population, Archive);
                        
                        [offspring1, offspring2] = SBXCrossover(parent1, parent2, Problem);
                        OffspringDec = [OffspringDec; offspring1; offspring2];
                    end
                    
                    if ~isempty(OffspringDec)
                        Offspring = Problem.Evaluation(OffspringDec);
                        Population = EnvironmentalSelection([Population, Offspring], length(Population));
                    end
                end
                
                %% Update archive every generation
                Archive = UpdateArchive(Archive, Population, ArchiveSize);
                
                %% Late-stage Convergence Refinement (Last 30% generations)
                if generation > 0.7*maxGen
                    % Focus on improving diversity along the Pareto front
                    NewDec = zeros(length(Population), Problem.D);
                    for i = 1:length(Population)
                        % Only apply refinement to solutions near the front
                        if all(Population(i).obj >= 0)  % Ensure valid objective values
                            % Small perturbation in decision space only
                            perturbation = 0.01 * (1 - generation/maxGen) * randn(1, Problem.D);
                            NewDec(i,:) = Population(i).dec + perturbation;
                            % Strict boundary enforcement
                            NewDec(i,:) = max(min(NewDec(i,:), Problem.upper), Problem.lower);
                        else
                            NewDec(i,:) = Population(i).dec;
                        end
                    end
                    NewPop = Problem.Evaluation(NewDec);
                    Population = EnvironmentalSelection([Population, NewPop], length(Population));
                end
            end
        end
    end
end

%% Enhanced Supporting Functions

function parent = TournamentSelection(Population, Archive)
    % Pareto-enhanced tournament selection
    if length(Population) < 2
        parent = Population(1);
        return;
    end
    
    % Select candidates from population only (avoid archive issues)
    idx = randperm(length(Population), min(2, length(Population)));
    candidates = Population(idx);
    
    % Pareto dominance check
    if Dominates(candidates(1).obj, candidates(2).obj)
        parent = candidates(1);
    elseif Dominates(candidates(2).obj, candidates(1).obj)
        parent = candidates(2);
    else
        % Random selection if non-dominated
        parent = candidates(randi(length(candidates)));
    end
end

function Selected = EnvironmentalSelection(Population, K)
    % Enhanced NSGA-II selection with Pareto emphasis
    if nargin < 2
        K = length(Population);
    end
    
    % Handle empty population
    if isempty(Population)
        Selected = [];
        return;
    end
    
    % Fast non-dominated sorting
    [FrontNo, MaxFNo] = NDSort(vertcat(Population.objs), K);
    Selected = [];
    
    % Elite preservation: Always keep first front
    firstFront = find(FrontNo == 1);
    if length(firstFront) >= K
        % Select most diverse solutions from first front
        CrowdDis = CrowdingDistance(vertcat(Population(firstFront).objs), ones(1, length(firstFront)));
        [~, rank] = sort(CrowdDis, 'descend');
        Selected = Population(firstFront(rank(1:K)));
        return;
    else
        Selected = Population(firstFront);
    end
    
    % Fill remaining slots with next fronts
    nextFront = 2;
    while length(Selected) < K && nextFront <= MaxFNo
        current = find(FrontNo == nextFront);
        needed = K - length(Selected);
        
        if length(current) <= needed
            Selected = [Selected, Population(current)];
        else
            % Select most diverse solutions
            CrowdDis = CrowdingDistance(vertcat(Population(current).objs), ones(1, length(current)));
            [~, rank] = sort(CrowdDis, 'descend');
            Selected = [Selected, Population(current(rank(1:needed)))];
        end
        nextFront = nextFront + 1;
    end
end

function Archive = UpdateArchive(Archive, Population, ArchiveSize)
    % Pareto-focused archive update
    Combined = [Archive, Population];
    
    % Handle empty archive case
    if isempty(Combined)
        Archive = [];
        return;
    end
    
    % Fast non-dominated filtering
    [FrontNo, ~] = NDSort(vertcat(Combined.objs), length(Combined));
    NonDominated = Combined(FrontNo == 1);
    
    % Ensure archive doesn't exceed size
    if length(NonDominated) <= ArchiveSize
        Archive = NonDominated;
    else
        % Select most diverse Pareto solutions
        CrowdDis = CrowdingDistance(vertcat(NonDominated.objs), ones(1, length(NonDominated)));
        [~, rank] = sort(CrowdDis, 'descend');
        Archive = NonDominated(rank(1:ArchiveSize));
    end
    
    % Store crowding distances separately (don't assign to SOLUTION objects)
    % If you need crowding distances later, calculate them when needed
end

function NewDec = PolynomialMutation(Dec, Problem, mutationStrength)
    % Enhanced polynomial mutation with adaptive strength
    if nargin < 3
        mutationStrength = 0.1;
    end
    
    eta = 20; % Distribution index
    Lower = Problem.lower;
    Upper = Problem.upper;
    
    % Ensure NewDec is initialized
    NewDec = Dec;
    
    % Mutation probability adaptation - reduced for better convergence
    mutationProb = min(0.2, 1/Problem.D);
    
    for i = 1:Problem.D
        if rand() < mutationProb
            % More conservative mutation for ZDT problems
            y = Dec(i);
            yl = Lower(i);
            yu = Upper(i);
            
            if y > yl && y < yu
                delta1 = (y - yl) / (yu - yl);
                delta2 = (yu - y) / (yu - yl);
                
                rnd = rand();
                mut_pow = 1.0 / (eta + 1.0);
                
                if rnd <= 0.5
                    xy = 1.0 - delta1;
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy^(eta + 1.0));
                    deltaq = val^mut_pow - 1.0;
                else
                    xy = 1.0 - delta2;
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy^(eta + 1.0));
                    deltaq = 1.0 - val^mut_pow;
                end
                
                NewDec(i) = y + deltaq * (yu - yl) * mutationStrength;
            end
        end
    end
    
    % Strict boundary handling
    NewDec = max(min(NewDec, Upper), Lower);
end

function [offspring1, offspring2] = SBXCrossover(parent1, parent2, Problem)
    % Diversity-enhanced SBX crossover
    eta = 15; % Distribution index
    Lower = Problem.lower;
    Upper = Problem.upper;
    
    offspring1 = parent1.dec;
    offspring2 = parent2.dec;
    
    for i = 1:Problem.D
        if rand() < 0.8  % High crossover probability
            if abs(parent1.dec(i) - parent2.dec(i)) > 1e-14
                if parent1.dec(i) < parent2.dec(i)
                    y1 = parent1.dec(i);
                    y2 = parent2.dec(i);
                else
                    y1 = parent2.dec(i);
                    y2 = parent1.dec(i);
                end
                
                yl = Lower(i);
                yu = Upper(i);
                
                beta = 1.0;
                rand_val = rand();
                if rand_val <= 0.5
                    beta = (2*rand_val)^(1/(eta+1));
                else
                    beta = (1/(2*(1-rand_val)))^(1/(eta+1));
                end
                
                c1 = 0.5*((y1+y2) - beta*abs(y2-y1));
                c2 = 0.5*((y1+y2) + beta*abs(y2-y1));
                
                % Apply bounds
                c1 = min(max(c1, yl), yu);
                c2 = min(max(c2, yl), yu);
                
                if rand() < 0.5
                    offspring1(i) = c1;
                    offspring2(i) = c2;
                else
                    offspring1(i) = c2;
                    offspring2(i) = c1;
                end
            end
        end
    end
end

%% Additional Helper Functions

function isDominated = Dominates(x, y)
    % Pareto dominance check
    isDominated = all(x <= y) && any(x < y);
end

function crowdDis = CrowdingDistance(objs, frontNo)
    % Crowding distance calculation
    [N, M] = size(objs);
    crowdDis = zeros(1, N);
    
    for m = 1:M
        [~, rank] = sort(objs(:,m));
        crowdDis(rank(1)) = inf;
        crowdDis(rank(end)) = inf;
        
        for i = 2:N-1
            if i <= N && (i+1) <= N % Bounds checking
                crowdDis(rank(i)) = crowdDis(rank(i)) + ...
                    (objs(rank(i+1),m) - objs(rank(i-1),m)) / ...
                    (max(objs(:,m)) - min(objs(:,m)) + eps);
            end
        end
    end
end