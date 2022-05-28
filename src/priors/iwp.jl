
struct IWP
    wiener_process_dimension::Int
    num_derivatives::Int
end


function discretize(iwp::IWP, h::Real)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives
    D = d * (q + 1)

    A, Q = zeros(D, D), zeros(D, D)

    @fastmath function fill_A!(A, h::Real)
        A .= 0
        for i in 1:D
            A[i,i] = 1
        end
        val = one(h)
        for i in 1:q
            val = val * h / i
            for k in 0:d-1
                for j in 1:q+1-i
                    @inbounds A[j+k*(q+1), j+k*(q+1)+i] = val
                end
            end
        end
    end

    @fastmath function _transdiff_ibm_element(row::Int, col::Int, h::Real)
        idx = 2 * q + 1 - row - col
        fact_rw = factorial(q - row)
        fact_cl = factorial(q - col)
        return h^idx / (idx * fact_rw * fact_cl)
    end
    @fastmath function fill_Q!(Q, h::Real)
        Q .= 0
        val = one(h)
        @simd for col in 0:q
            @simd for row in col:q
                val = _transdiff_ibm_element(row, col, h)
                @simd for i in 0:d-1
                    @inbounds Q[1+col+i*(q+1), 1+row+i*(q+1)] = val
                    @inbounds Q[1+row+i*(q+1), 1+col+i*(q+1)] = val
                end
            end
        end
    end

    fill_A!(A, h)
    fill_Q!(Q, h)

    return A, Q
end

function projectionmatrix(iwp::IWP, derivative::Integer)
    d = iwp.wiener_process_dimension
    q = iwp.num_derivatives
    return kron(diagm(0 => ones(d)), [i==(derivative+1) ? 1 : 0 for i in 1:q+1]')
end
