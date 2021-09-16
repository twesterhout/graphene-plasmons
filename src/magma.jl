using Libdl
using CEnum
using LinearAlgebra

const dependency_lock = ReentrantLock()

# lazily initialize a Ref containing a path to a library.
# the arguments to this macro is the name of the ref, an expression to populate it
# (possibly returning `nothing` if the library wasn't found), and an optional initialization
# hook to be executed after successfully discovering the library and setting the ref.
macro initialize_ref(ref, ex, hook=:())
    quote
        ref = $ref

        # test and test-and-set
        if !isassigned(ref)
            Base.@lock dependency_lock begin
                if !isassigned(ref)
                    val = $ex
                    if val === nothing && !(eltype($ref) <: Union{Nothing,<:Any})
                        error($"Could not find a required library")
                    end
                    $ref[] = val
                    if val !== nothing
                        $hook
                    end
                end
            end
        end

        $ref[]
    end
end

const __libmagma = Ref{String}()
function libmagma()
    @initialize_ref __libmagma begin
        find_library("libmagma")
    end
end
hasmagma() = !isempty(libmagma())

const magma_int_t = Cint

@cenum magma_vec_t::UInt begin
    MagmaNoVec = 301
    MagmaVec = 302
    MagmaIVec = 303
    MagmaAllVec = 304
    MagmaSomeVec = 305
    MagmaOverwriteVec = 306
    MagmaBacktransVec = 307
end

function magma_init()
    ccall((:magma_init, libmagma()), magma_int_t, ())
end

function magma_finalize()
    ccall((:magma_finalize, libmagma()), magma_int_t, ())
end

function magma_geev()

end

# function chkfinite(A::AbstractMatrix)
#     for a in A
#         if !isfinite(a)
#             throw(ArgumentError("matrix contains Infs or NaNs"))
#         end
#     end
#     return true
# end

@inline function char_to_vec(op::AbstractChar)
    op == 'V' && return MagmaVec
    op == 'N' && return MagmaNoVec
    throw(ArgumentError("invalid op: $op; expected either 'N' or 'V'"))
end

for (fname, elty) in ((Symbol(":magma_zgeev"), :ComplexF64), ) # ("magma_cgeev", :ComplexF32), 
    @eval begin
        function magma_geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty})
            stride(A, 1) == 1 || throw(ArgumentError("matrix A does not have contiguous columns"))
            n = size(A, 1)
            size(A, 1) == size(A, 2) || throw(ArgumentError("matrix A is not square"))
            # chkfinite(A) # balancing routines don't support NaNs and Infs
            lvecs = jobvl == 'V'
            rvecs = jobvr == 'V'
            VL    = similar(A, $elty, (n, lvecs ? n : 0))
            VR    = similar(A, $elty, (n, rvecs ? n : 0))
            w     = similar(A, $elty, n)
            rwork = similar(A, real($elty), 2 * n)
            work  = Vector{$elty}(undef, 1)
            lwork = magma_int_t(-1)
            info  = Ref{magma_int_t}()
            lda   = max(1, stride(A, 2))

            jobvl = char_to_vec(jobvl)
            jobvr = char_to_vec(jobvr)

            for i = 1:2  # first call returns lwork as work[1]
                if $elty == ComplexF64
                    ccall((:magma_zgeev, libmagma()), magma_int_t,
                          (magma_vec_t, magma_vec_t, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, Ref{$elty}, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, magma_int_t, Ref{real($elty)}, Ref{magma_int_t}),
                          jobvl, jobvr, n, A, lda, w, VL, n, VR, n, work, lwork, rwork, info)
                else
                    @assert $elty == ComplexF32
                    ccall((:magma_cgeev, libmagma()), magma_int_t,
                          (magma_vec_t, magma_vec_t, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, Ref{$elty}, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, magma_int_t, Ref{real($elty)}, Ref{magma_int_t}),
                          jobvl, jobvr, n, A, lda, w, VL, n, VR, n, work, lwork, rwork, info)
                end
                if i == 1
                    lwork = ceil(magma_int_t, real(work[1]))
                    resize!(work, lwork)
                end
            end
            w, VL, VR
        end

        function magma_geev_m!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty})
            stride(A, 1) == 1 || throw(ArgumentError("matrix A does not have contiguous columns"))
            n = size(A, 1)
            size(A, 1) == size(A, 2) || throw(ArgumentError("matrix A is not square"))
            # chkfinite(A) # balancing routines don't support NaNs and Infs
            lvecs = jobvl == 'V'
            rvecs = jobvr == 'V'
            VL    = similar(A, $elty, (n, lvecs ? n : 0))
            VR    = similar(A, $elty, (n, rvecs ? n : 0))
            w     = similar(A, $elty, n)
            rwork = similar(A, real($elty), 2 * n)
            work  = Vector{$elty}(undef, 1)
            lwork = magma_int_t(-1)
            info  = Ref{magma_int_t}()
            lda   = max(1, stride(A, 2))

            jobvl = char_to_vec(jobvl)
            jobvr = char_to_vec(jobvr)

            for i = 1:2  # first call returns lwork as work[1]
                if $elty == ComplexF64
                    ccall((:magma_zgeev_m, libmagma()), magma_int_t,
                          (magma_vec_t, magma_vec_t, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, Ref{$elty}, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, magma_int_t, Ref{real($elty)}, Ref{magma_int_t}),
                          jobvl, jobvr, n, A, lda, w, VL, n, VR, n, work, lwork, rwork, info)
                else
                    @assert $elty == ComplexF32
                    ccall((:magma_cgeev_m, libmagma()), magma_int_t,
                          (magma_vec_t, magma_vec_t, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, Ref{$elty}, magma_int_t, Ref{$elty}, magma_int_t,
                           Ref{$elty}, magma_int_t, Ref{real($elty)}, Ref{magma_int_t}),
                          jobvl, jobvr, n, A, lda, w, VL, n, VR, n, work, lwork, rwork, info)
                end
                if i == 1
                    lwork = ceil(magma_int_t, real(work[1]))
                    resize!(work, lwork)
                end
            end
            w, VL, VR
        end
    end
end
