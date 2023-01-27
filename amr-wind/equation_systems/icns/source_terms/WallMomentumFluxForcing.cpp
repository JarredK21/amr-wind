#include "amr-wind/equation_systems/icns/source_terms/WallMomentumFluxForcing.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/FieldUtils.H"
#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/wind_energy/ShearStress.H"

#include "AMReX_ParmParse.H"

namespace amr_wind::pde::icns {

// FIXME: comments out of date
/** Boussinesq buoyancy source term for ABL simulations
 *
 *  Reads in the following parameters from `ABLMeanBoussinesq` namespace:
 *
 *  - `reference_temperature` (Mandatory) temperature (`T0`) in Kelvin
 *  - `thermal_expansion_coeff` Optional, default = `1.0 / T0`
 *  - `gravity` acceleration due to gravity (m/s)
 *  - `read_temperature_profile`
 *  - `tprofile_filename`
 */
WallMomentumFluxForcing::WallMomentumFluxForcing(const CFDSim& sim)
    : m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_density(sim.repo().get_field("density"))
    , m_mo(sim.physics_manager().get<amr_wind::ABL>().abl_wall_function().mo())
{

    // some parm parse stuff?
    amrex::ParmParse pp("ABL");
    pp.query("wall_shear_stress_type", m_wall_shear_stress_type);
    pp.query("normal_direction", m_direction);
    AMREX_ASSERT((0 <= m_direction) && (m_direction < AMREX_SPACEDIM));
}

WallMomentumFluxForcing::~WallMomentumFluxForcing() = default;

void WallMomentumFluxForcing::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
    // Overall geometry information
    const auto& geom = m_mesh.Geom(lev);

    // Mesh cell size information
    const auto& dx = m_mesh.Geom(lev).CellSizeArray();
    amrex::Real dV = 1.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        dV *= dx[dir];
    }

    // Domain size information.
    const auto& domain = geom.Domain();

    //
    const int idir = m_direction;

    const auto& velocityField = m_velocity.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    FieldState densityState = field_impl::phi_state(fstate);
    const auto& density = m_density.state(densityState)(lev).const_array(mfi);

    if (!(bx.smallEnd(idir) == domain.smallEnd(idir))) return;
    if (idir != 2) return;

    amrex::ParallelFor(
        amrex::bdryLo(bx, idir),
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

            // Get the local velocity at the cell center adjacent
            // to this wall face.
            const amrex::Real u = velocityField(i, j, k, 0);
            const amrex::Real v = velocityField(i, j, k, 1);
            const amrex::Real S = std::sqrt((u*u) + (v*v));


            // Get local tau_wall based on the local conditions and
            // mean state based on Monin-Obukhov similarity.
            amrex::Real tau_xz = 0.0;
            amrex::Real tau_yz = 0.0;

            if (m_wall_shear_stress_type == "constant") {
                auto tau = ShearStressConstant(m_mo);
                tau_xz = tau.calc_vel_x(u,S);
                tau_yz = tau.calc_vel_y(v,S);
            } else if (m_wall_shear_stress_type == "default") {
                auto tau = ShearStressDefault(m_mo);
                tau_xz = tau.calc_vel_x(u,S);
                tau_yz = tau.calc_vel_y(v,S);
            } else if (m_wall_shear_stress_type == "local") {
                auto tau = ShearStressLocal(m_mo);
                tau_xz = tau.calc_vel_x(u,S);
                tau_yz = tau.calc_vel_y(v,S);
            } else if (m_wall_shear_stress_type == "schumann") {
                auto tau = ShearStressSchumann(m_mo);
                tau_xz = tau.calc_vel_x(u,S);
                tau_yz = tau.calc_vel_y(v,S);
            } else {
                auto tau = ShearStressMoeng(m_mo);
                tau_xz = tau.calc_vel_x(u,S);
                tau_yz = tau.calc_vel_y(v,S);
            }


            // Adding the source term as surface stress vector times surface area divided by cell
            // volume (division by cell volume is to make this a source per unit volume).
            src_term(i, j, k, 0) -= (tau_xz * dx[0] * dx[1]) / dV;
            src_term(i, j, k, 1) -= (tau_yz * dx[1] * dx[1]) / dV;
            src_term(i, j, k, 2) += 0.0;
          //src_term(i, j, k, 0) -= 0.1*velocityField(i, j, k, 0);
          //src_term(i, j, k, 1) += 0.2*velocityField(i, j, k, 1);
          //src_term(i, j, k, 2) += 0.0;
        });
}

} // namespace amr_wind::pde::icns