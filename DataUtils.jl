my_display(x) = display(x), println()

function DisplayTxtHeads(input_path::AbstractString, num_line::Int)
    io = open(input_path, "r")
    for _ in 1:num_line
        println(readline(io; keep=false))
    end
    close(io)
end

DisplayTxtHeads("data/roadNet-TX.txt", 10)