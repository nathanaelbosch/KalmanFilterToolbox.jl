
projectionmatrix(d::Integer, q::Integer, derivative::Integer) =
    kron(diagm(0 => ones(d)), [i == (derivative + 1) ? 1 : 0 for i in 1:q+1]')

"""
    projectionmatrix(process, derivative::Integer)

Compute the projection matrix that maps the state to the specified derivative.
"""
function projectionmatrix(process, derivative::Integer)
    d = process.wiener_process_dimension
    q = process.num_derivatives
    projectionmatrix(d, q, derivative)
end
