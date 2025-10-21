module VectReverse

mutable struct VectNode
	op::Union{Nothing, Symbol}
    args::Vector{VectNode}
    value::AbstractArray
    derivative::AbstractArray
end

VectNode(op, args, value) = VectNode(op, args, value, zeros(size(value)))
VectNode(value) = VectNode(nothing, VectNode[], value)

# For `tanh.(X)`
function Base.broadcasted(op::Function, x::VectNode)
    if op == tanh
        return VectNode(Symbol("tanh."), [x], tanh.(x.value))
    elseif op == exp
        return VectNode(Symbol("exp."), [x], exp.(x.value))
    elseif op == log
        return VectNode(Symbol("log."), [x], log.(x.value))
	elseif op == -
        return VectNode(Symbol(".-"), [x], log.(x.value))
    else
        error("Unsupported broadcasted unary op: $op")
    end
end

# For `X .* Y`
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    if op == *
        return VectNode(Symbol(".*"), [x, y], x.value .* y.value)
    elseif op == +
        return VectNode(Symbol(".+"), [x, y], x.value .+ y.value)
    elseif op == -
        return VectNode(Symbol(".-"), [x, y], x.value .- y.value)
    elseif op == /
        return VectNode(Symbol("./"), [x, y], x.value ./ y.value)
    else
        error("Unsupported broadcasted op: $op")
    end
end


x1 = VectNode([5])
log.(x1)
x2 = VectNode([6])

x1 ./ x2

a1 = [1 2 3;4 5 6]
a1 .= 0

# For `X .* Y` where `Y` is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    if op == *
        return x .* VectNode(y)
    elseif op == +
        return x .+ VectNode(y)
    elseif op == -
        return x .- VectNode(y)
    elseif op == /
        return x ./ VectNode(y)
    else
        error("Unsupported broadcasted op: $op")
    end
end

x1 = VectNode([5])
x2 = [6]

x1 ./ x2

# For `X .* Y` where `X` is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    if op == *
        return VectNode(x) .* y
    elseif op == +
        return VectNode(x) .+ y
    elseif op == -
        return VectNode(x) .- y
    elseif op == /
        return VectNode(x) ./ y
    else
        error("Unsupported broadcasted op: $op")
    end
end

x1 = Float64[1 2 3; 4 5 6]
x2 = VectNode(Float64[1 2 3; 4 5 6])

# For `x .^ 2`
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
	Base.broadcasted(^, x, y)
end

# We assume `Flatten` has been defined in the parent module.
# If this fails, run `include("/path/to/Flatten.jl")` before
# including this file.
include(joinpath(@__DIR__, "../LabAD/flatten.jl"))
import ..Flatten

function topo_sort!(visited, topo, f::VectNode)
	if !(f in visited)
		push!(visited, f)
		for arg in f.args
			topo_sort!(visited, topo, arg)
		end
		push!(topo, f)
	end
end

function _backward!(f::VectNode)
	if isnothing(f.op)
		return
	elseif f.op == Symbol(".+")
		for arg in f.args
			arg.derivative .+= f.derivative
		end
	elseif f.op == Symbol(".-") && length(f.args) == 2
		f.args[1].derivative .+= f.derivative
		f.args[2].derivative .-= f.derivative
	elseif f.op == Symbol(".-") && length(f.args) == 1
		f.args[1].derivative .-= f.derivative
	elseif f.op == Symbol(".*") && length(f.args) == 2
		f.args[1].derivative .+= f.derivative .* f.args[2].value
		f.args[2].derivative .+= f.derivative .* f.args[1].value
	elseif f.op == Symbol("./") && length(f.args) == 2
		f.args[1].derivative .+= f.derivative ./ f.args[2].value
		f.args[2].derivative .-= f.derivative .* f.args[1].value ./ (f.args[2].value .^ 2)
	elseif f.op == Symbol("tanh.") && length(f.args) == 1
		f.args[1].derivative .+= f.derivative .* (1 .- tanh.(f.args[1].value).^2)
	elseif f.op == Symbol("exp.") && length(f.args) == 1
		f.args[1].derivative .+= f.derivative .* exp.(f.args[1].value)
	elseif f.op == Symbol("log.") && length(f.args) == 1
		f.args[1].derivative .+= f.derivative ./ f.args[1].value
	else
		error("Operator $(f.op) not supported yet")
	end
end

function backward!(f::VectNode)
	topo = typeof(f)[]
	topo_sort!(Set{typeof(f)}(), topo, f)
	reverse!(topo)
	for node in topo
		node.derivative .= 0
	end
	f.derivative .= 1
	for node in topo
		_backward!(node)
	end
	return f
end

function gradient!(f, g::Flatten, x::Flatten)
	x_nodes = Flatten(VectNode.(x.components))
	expr = f(x_nodes)
	backward!(expr)
	for i in eachindex(x.components)
		g.components[i] .= x_nodes.components[i].derivative
	end
	return g
end

gradient(f, x) = gradient!(f, zero(x), x)



end