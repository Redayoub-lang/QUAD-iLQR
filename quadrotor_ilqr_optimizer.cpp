#include 
#include 
#include 
#include 
#include 

namespace quad_ilqr {

constexpr int STATE_DIM = 12; // [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
constexpr int CONTROL_DIM = 4; // [Thrust, Torque_x, Torque_y, Torque_z]

using StateVector = Eigen::Matrix;
using ControlVector = Eigen::Matrix;
using StateMatrix = Eigen::Matrix;
using ControlMatrix = Eigen::Matrix;
using BMatrix = Eigen::Matrix;

struct QuadrotorParams {
    double mass = 1.5;         // kg
    double gravity = 9.81;     // m/s^2
    Eigen::Vector3d I = {0.015, 0.015, 0.03}; // Inertia moments
};

struct Obstacle {
    Eigen::Vector3d center;
    double radius;
    double weight = 100.0;
};

class QuadrotorILQR {
public:
    QuadrotorILQR(int horizon, double dt, QuadrotorParams params)
        : N_(horizon), dt_(dt), params_(params) {
        
        Q_.setIdentity(); Q_ *= 10.0;
        Q_.block<3, 3>(0, 0) *= 5.0;
        R_.setIdentity(); R_ *= 0.1;
        Qf_ = Q_ * 2.0;

        u_sequence_.assign(N_, ControlVector::Zero());
        for (auto& u : u_sequence_) u(0) = params_.mass * params_.gravity;
    }

    StateVector continuous_dynamics(const StateVector& x, const ControlVector& u) const {
        StateVector x_dot;
        const double phi = x(6), theta = x(7), psi = x(8);
        const double p = x(9), q = x(10), r = x(11);

        x_dot.segment<3>(0) = x.segment<3>(3);

        Eigen::Matrix3d R;
        R << std::cos(theta)*std::cos(psi), std::sin(phi)*std::sin(theta)*std::cos(psi) - std::cos(phi)*std::sin(psi), std::cos(phi)*std::sin(theta)*std::cos(psi) + std::sin(phi)*std::sin(psi),
             std::cos(theta)*std::sin(psi), std::sin(phi)*std::sin(theta)*std::sin(psi) + std::cos(phi)*std::cos(psi), std::cos(phi)*std::sin(theta)*std::sin(psi) - std::sin(phi)*std::cos(psi),
            -std::sin(theta),               std::sin(phi)*std::cos(theta),                                             std::cos(phi)*std::cos(theta);

        Eigen::Vector3d thrust_body(0.0, 0.0, u(0));
        x_dot.segment<3>(3) = Eigen::Vector3d(0, 0, -params_.gravity) + (R * thrust_body) / params_.mass;

        Eigen::Matrix3d W;
        W << 1.0, std::sin(phi)*std::tan(theta), std::cos(phi)*std::tan(theta),
             0.0, std::cos(phi),               -std::sin(phi),
             0.0, std::sin(phi)/std::cos(theta), std::cos(phi)/std::cos(theta);
        x_dot.segment<3>(6) = W * x.segment<3>(9);

        Eigen::Vector3d torques = u.segment<3>(1);
        x_dot(9)  = (torques.x() - (params_.I.z() - params_.I.y()) * q * r) / params_.I.x();
        x_dot(10) = (torques.y() - (params_.I.x() - params_.I.z()) * p * r) / params_.I.y();
        x_dot(11) = (torques.z() - (params_.I.y() - params_.I.x()) * p * q) / params_.I.z();

        return x_dot;
    }

    StateVector rk4_step(const StateVector& x, const ControlVector& u) const {
        StateVector k1 = continuous_dynamics(x, u);
        StateVector k2 = continuous_dynamics(x + 0.5 * dt_ * k1, u);
        StateVector k3 = continuous_dynamics(x + 0.5 * dt_ * k2, u);
        StateVector k4 = continuous_dynamics(x + dt_ * k3, u);
        return x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    void add_obstacle(const Obstacle& obs) { obstacles_.push_back(obs); }

    double compute_cost(const std::vector& x_seq, const std::vector& u_seq, const StateVector& x_ref) const {
        double total_cost = 0.0;
        ControlVector u_hover = ControlVector::Zero();
        u_hover(0) = params_.mass * params_.gravity;

        for (int k = 0; k < N_; ++k) {
            StateVector e_x = x_seq[k] - x_ref;
            ControlVector e_u = u_seq[k] - u_hover;
            total_cost += 0.5 * e_x.transpose() * Q_ * e_x;
            total_cost += 0.5 * e_u.transpose() * R_ * e_u;

            Eigen::Vector3d pos = x_seq[k].segment<3>(0);
            for (const auto& obs : obstacles_) {
                double dist_sq = (pos - obs.center).squaredNorm();
                total_cost += obs.weight * std::exp(-dist_sq / (2.0 * obs.radius * obs.radius));
            }
        }
        StateVector e_xf = x_seq.back() - x_ref;
        total_cost += 0.5 * e_xf.transpose() * Qf_ * e_xf;
        return total_cost;
    }

    void solve(const StateVector& x0, const StateVector& x_ref, int max_iters = 30) {
        std::vector x_seq(N_ + 1, x0);
        for (int k = 0; k < N_; ++k) {
            x_seq[k + 1] = rk4_step(x_seq[k], u_sequence_[k]);
        }

        double current_cost = compute_cost(x_seq, u_sequence_, x_ref);
        double mu = 1e-6;

        for (int iter = 0; iter < max_iters; ++iter) {
            std::vector> K_gains(N_);
            std::vector k_feedforward(N_);

            StateVector Vx = Qf_ * (x_seq.back() - x_ref);
            StateMatrix Vxx = Qf_;

            bool backward_success = true;
            for (int k = N_ - 1; k >= 0; --k) {
                StateMatrix A;
                BMatrix B;
                compute_jacobians(x_seq[k], u_sequence_[k], A, B);

                StateVector lx = Q_ * (x_seq[k] - x_ref);
                ControlVector lu = R_ * (u_sequence_[k] - ControlVector(params_.mass * params_.gravity, 0, 0, 0));

                StateVector Qx = lx + A.transpose() * Vx;
                ControlVector Qu = lu + B.transpose() * Vx;
                StateMatrix Qxx = Q_ + A.transpose() * Vxx * A;
                ControlMatrix Quu = R_ + B.transpose() * Vxx * B + mu * ControlMatrix::Identity();
                Eigen::Matrix Qux = B.transpose() * Vxx * A;

                Eigen::LLT llt(Quu);
                if (llt.info() == Eigen::NumericalIssue) {
                    backward_success = false;
                    break;
                }

                k_feedforward[k] = -llt.solve(Qu);
                K_gains[k] = -llt.solve(Qux);

                Vx = Qx + K_gains[k].transpose() * Quu * k_feedforward[k] + K_gains[k].transpose() * Qu + Qux.transpose() * k_feedforward[k];
                Vxx = Qxx + K_gains[k].transpose() * Quu * K_gains[k] + K_gains[k].transpose() * Qux + Qux.transpose() * K_gains[k];
            }

            if (!backward_success) {
                mu *= 10.0;
                continue;
            }

            double alpha = 1.0;
            bool cost_improved = false;
            std::vector x_new(N_ + 1, x0);
            std::vector u_new = u_sequence_;

            for (int line_search = 0; line_search < 10; ++line_search) {
                for (int k = 0; k < N_; ++k) {
                    ControlVector du = alpha * k_feedforward[k] + K_gains[k] * (x_new[k] - x_seq[k]);
                    u_new[k] = u_sequence_[k] + du;
                    x_new[k + 1] = rk4_step(x_new[k], u_new[k]);
                }

                double new_cost = compute_cost(x_new, u_new, x_ref);
                if (new_cost < current_cost) {
                    current_cost = new_cost;
                    x_seq = x_new;
                    u_sequence_ = u_new;
                    cost_improved = true;
                    mu = std::max(1e-6, mu / 5.0);
                    break;
                }
                alpha *= 0.5;
            }

            if (!cost_improved) {
                mu *= 5.0;
            }

            std::cout << "[iLQR Iter " << iter << "] Cost: " << current_cost << " | Regularization (mu): " << mu << "\n";
            if (alpha < 1e-4) break;
        }
    }

private:
    void compute_jacobians(const StateVector& x, const ControlVector& u, StateMatrix& A, BMatrix& B) const {
        const double eps = 1e-6;
        StateVector f0 = rk4_step(x, u);

        for (int i = 0; i < STATE_DIM; ++i) {
            StateVector x_inc = x;
            x_inc(i) += eps;
            A.col(i) = (rk4_step(x_inc, u) - f0) / eps;
        }

        for (int j = 0; j < CONTROL_DIM; ++j) {
            ControlVector u_inc = u;
            u_inc(j) += eps;
            B.col(j) = (rk4_step(x, u_inc) - f0) / eps;
        }
    }

    int N_;
    double dt_;
    QuadrotorParams params_;
    StateMatrix Q_, Qf_;
    ControlMatrix R_;
    std::vector u_sequence_;
    std::vector obstacles_;
};

} // namespace quad_ilqr

int main() {
    std::cout << "===========================================================\n";
    std::cout << "  QUAD-iLQR: Optimal Trajectory Optimizer & Obstacle Field \n";
    std::cout << "===========================================================\n";

    quad_ilqr::QuadrotorParams params;
    quad_ilqr::QuadrotorILQR solver(50, 0.05, params);

    solver.add_obstacle({Eigen::Vector3d(1.5, 1.5, 2.0), 0.8, 250.0});

    quad_ilqr::StateVector x0 = quad_ilqr::StateVector::Zero();
    quad_ilqr::StateVector x_ref = quad_ilqr::StateVector::Zero();
    x_ref.segment<3>(0) = Eigen::Vector3d(3.0, 3.0, 4.0);

    std::cout << "Optimizing Trajectory from [0,0,0] to [3,3,4] with Obstacle at [1.5,1.5,2]...\n\n";
    solver.solve(x0, x_ref, 20);

    std::cout << "\nOptimization Complete. Trajectory successfully optimized!\n";
    return 0;
}