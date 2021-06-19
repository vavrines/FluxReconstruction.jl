using PyCall

py"""
import numpy
def get_phifj_solution_grad_tri(order, P1, Np, Nflux, elem_nfaces, V, Vf):

    correction_coeffs = numpy.zeros( shape=(Np, elem_nfaces*P1) )

    phifj_solution_r = numpy.zeros( shape=(Np, Nflux) )
    phifj_solution_s = numpy.zeros( shape=(Np, Nflux) )

    tmp = 1./numpy.sqrt(2)

    nhat = numpy.zeros(shape=(Nflux,2))
    nhat[0:P1, 0] =  0.0
    nhat[0:P1, 1] = -1.0

    nhat[P1:2*P1, :] = tmp

    nhat[2*P1:, 0] = -1
    nhat[2*P1:, 1] = 0.0

    #wgauss, rgauss = get_gauss_nodes(order)
    rgauss, wgauss = numpy.polynomial.legendre.leggauss(order+1)

    for m in range(Np):
        for face in range(3):
            modal_basis_along_face = Vf[ face*P1:(face+1)*P1, m ]
            correction_coeffs[m, face*P1:(face+1)*P1] = modal_basis_along_face*wgauss

    # correct the coefficients for face 2 with the Hypotenuse length
    correction_coeffs[:, P1:2*P1] *= numpy.sqrt(2)

    # Multiply the correction coefficients with the Dubiner basis
    # functions evaluated at the solution and flux points to get the
    # correction functions
    phifj_solution = V.dot(correction_coeffs)

    # multiply each row of the correction function with the
    # transformed element normals. These matrices will be used to
    # compute the gradients and flux divergence
    for m in range(Np):
        phifj_solution_r[m, :] = phifj_solution[m, :] * nhat[:, 0]
        phifj_solution_s[m, :] = phifj_solution[m, :] * nhat[:, 1]

    # stack the matrices
    phifj_grad = numpy.zeros( shape=(3*Np, Nflux) )
    phifj_grad[:Np,     :] = phifj_solution_r[:, :]
    phifj_grad[Np:2*Np, :] = phifj_solution_s[:, :]

    return correction_coeffs, phifj_solution, phifj_grad
"""
