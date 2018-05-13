import torch
import itertools
from .aev_base import AEVComputer
from . import buildin_const_file, default_dtype, default_device
from . import _utils


class AEV(AEVComputer):
    """The AEV computer fully implemented using pytorch, making use of neighbor list

    Attributes
    ----------
    timers : dict
        Dictionary storing the the benchmark result. It has the following keys:
            neighborlist : time spent on computing neighborlist
            aev : time spent on computing AEV, after the nighborlist is given
            total : total time for computing everything, including neighborlist and AEV.
    """

    def __init__(self, benchmark=False, device=default_device, dtype=default_dtype, const_file=buildin_const_file):
        super(AEV, self).__init__(benchmark, dtype, device, const_file)
        if benchmark:
            self.radial_subaev = self._enable_benchmark(
                self.radial_subaev, 'radial_subaev')
            self.angular_subaev = self._enable_benchmark(
                self.angular_subaev, 'angular_subaev')
            self.compute_neighborlist = self._enable_benchmark(
                self.compute_neighborlist, 'neighborlist')
            self.compute_aev_using_neighborlist = self._enable_benchmark(
                self.compute_aev_using_neighborlist, 'aev')
            self.forward = self._enable_benchmark(self.forward, 'total')

    def radial_subaev(self, distances):
        """Compute the radial subAEV of the center atom given neighbors

        The radial AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 3.
        The sum computed by this method is over all given neighbors, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        distances : torch.Tensor
            Pytorch tensor of shape (conformations, N) storing the |Rij| length where i is the
            center atom, and j is a neighbor. The |Rij| of conformation n is stored as (n,j).

        Returns
        -------
        torch.Tensor
            A tensor of shape (conformations, `radial_sublength()`) storing the subAEVs.
        """
        # use broadcasting semantics to do Cartesian product on constants
        # shape convension (conformations, atoms, EtaR, ShfR)
        distances = distances.unsqueeze(-1).unsqueeze(-1)
        fc = AEVComputer._cutoff_cosine(distances, self.Rcr)
        eta = self.EtaR.view(1, 1, -1, 1)
        radius_shift = self.ShfR.view(1, 1, 1, -1)
        # Note that in the equation in the paper there is no 0.25 coefficient, but in NeuroChem there is such a coefficient. We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-eta * (distances - radius_shift)**2) * fc
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last two dimensions to view the subAEV as one dimensional vector
        return ret.view(-1, self.radial_sublength)

    def angular_subaev(self, Rij_vec):
        """Compute the angular subAEV of the center atom given neighbor pairs.

        The angular AEV is define in https://arxiv.org/pdf/1610.08935.pdf equation 4.
        The sum computed by this method is over all given neighbor pairs, so the caller
        of this method need to select neighbors if the caller want a per species subAEV.

        Parameters
        ----------
        Rij_vec : torch.Tensor
            Tensor of shape (conformations, N, 2, 3) storing the Rij vectors where i is the
            center atom, and j is a neighbor. The vector (n,k,l,:) is the Rij where j refer
            to the l-th atom of the k-th pair.

        Returns
        -------
        torch.Tensor
            Tensor of shape (conformations, `angular_sublength`) storing the subAEVs.
        """
        pairs = Rij_vec.shape[1]
        R_distances = torch.norm(Rij_vec, 2, dim=-1)
        """pytorch tensor of shape (conformations, N, 2) storing the |Rij| length where i is the
        center atom, and j is a neighbor. The value at (n,k,l) is the |Rij| where j refer to the
        l-th atom of the k-th pair."""

        # Compute the angles jik with i in the center and j and k are the two atoms in a pair.
        # The result tensor would have shape (conformations, pairs)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        cos_angles = 0.95 * \
            torch.nn.functional.cosine_similarity(
                Rij_vec[:, :, 0, :], Rij_vec[:, :, 1, :], dim=-1)
        angles = torch.acos(cos_angles)

        # use broadcasting semantics to combine constants
        # shape convension (conformations, pairs, EtaA, Zeta, ShfA, ShfZ)
        angles = angles.view(-1, pairs, 1, 1, 1, 1)
        Rij = R_distances.view(-1, pairs, 2, 1, 1, 1, 1)
        fcj = AEVComputer._cutoff_cosine(Rij, self.Rca)
        eta = self.EtaA.view(1, 1, -1, 1, 1, 1)
        zeta = self.Zeta.view(1, 1, 1, -1, 1, 1)
        radius_shifts = self.ShfA.view(1, 1, 1, 1, -1, 1)
        angle_shifts = self.ShfZ.view(1, 1, 1, 1, 1, -1)
        ret = 2 * ((1 + torch.cos(angles - angle_shifts)) / 2) ** zeta * \
            torch.exp(-eta * ((Rij[:, :, 0, :, :, :, :] + Rij[:, :, 1, :, :, :, :]) / 2 - radius_shifts)
                      ** 2) * fcj[:, :, 0, :, :, :, :] * fcj[:, :, 1, :, :, :, :]
        # end of shape convension
        ret = torch.sum(ret, dim=1)
        # flat the last 4 dimensions to view the subAEV as one dimension vector
        return ret.view(-1, self.angular_sublength)

    def compute_neighborlist(self, coordinates, species):
        """Compute neighbor list of each atom, and group neighbors by species

        Parameters
        ----------
        coordinates : pytorch tensor of `dtype`
            The tensor that specifies the xyz coordinates of atoms in the molecule.
            The tensor must have shape (conformations, atoms, 3)
        species : list of str
            The list that specifies the species of each atom. The length of the list
            must match with `coordinates.shape[1]`.

        Returns
        -------
        dict
            Dictionary storing neighbor information. The key for this dictionary is species,
            and the corresponding value is a list of size `atoms`. The elements in the list
            is a pair of long tensors storing the indices of neighbors of that atom.
        """
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]
        
        zero_radial_subaev = torch.zeros(conformations, self.radial_sublength, dtype=self.dtype, device=self.device)

        indices = {}
        for s in self.species:
            indices[s] = []

        masks = {}
        for s in self.species:
            mask = [1 if x == s else 0 for x in species]
            masks[s] = torch.tensor(mask, dtype=torch.uint8, device=self.device)

        radial_aevs = []
        """The list whose elements are full radial AEV of each atom"""

        for i in range(atoms):
            center = coordinates[:, i:i+1, :]
            R_vecs = coordinates - center
            R_distances = torch.norm(R_vecs, 2, dim=-1)

            in_Rcr = R_distances <= self.Rcr
            in_Rcr = torch.sum(in_Rcr.type(torch.float), dim=0) > 0
            in_Rcr[i] = 0

            in_Rca = R_distances <= self.Rca
            in_Rca = torch.sum(in_Rca.type(torch.float), dim=0) > 0
            in_Rca[i] = 0

            radial_aev = []
            """The list whose elements are atom i's per species subAEV of each species"""

            for s in self.species:
                mask = masks[s]

                # compute radial AEV
                in_Rcr_idx = (in_Rcr * mask).nonzero().view(-1)
                if in_Rcr_idx.shape[0] > 0:
                    radial_aev.append(self.radial_subaev(R_distances.index_select(1, in_Rcr_idx)))
                else:
                    radial_aev.append(zero_radial_subaev)

                # compute angular
                in_Rca_idx = (in_Rca * mask).nonzero().view(-1)
                indices[s].append(R_vecs.index_select(1, in_Rca_idx))

            radial_aev = torch.cat(radial_aev, dim=1)
            radial_aevs.append(radial_aev)

        radial_aevs = torch.stack(radial_aevs, dim=1)
        return radial_aevs, indices

    def compute_aev_using_neighborlist(self, coordinates, species, neighbor_indices):
        conformations = coordinates.shape[0]
        atoms = coordinates.shape[1]

        # helper exception for control flow
        class AEVIsZero(Exception):
            pass

        # compute angular AEV
        angular_aevs = []
        zero_subaev = torch.zeros(conformations, self.angular_sublength, dtype=self.dtype, device=self.device)
        """The list whose elements are full angular AEV of each atom"""
        for i in range(atoms):
            angular_aev = []
            """The list whose elements are atom i's per species subAEV of each species"""
            for j, k in itertools.combinations_with_replacement(self.species, 2):
                try:
                    R_vecs_j = neighbor_indices[j][i]
                    if len(R_vecs_j) == 0:
                        raise AEVIsZero()
                    if j != k:
                        # the two atoms in the pair have different species
                        R_vecs_k = neighbor_indices[k][i]
                        if len(R_vecs_k) == 0:
                            raise AEVIsZero()
                        R_vecs_pairs = _utils.cartesian_prod(R_vecs_j, R_vecs_k, dim=1, newdim=2)
                    else:
                        # the two atoms in the pair have the same species j
                        if len(R_vecs_j.shape) != 3 or R_vecs_j.shape[1] < 2:
                            raise AEVIsZero()
                        R_vecs_pairs = _utils.combinations(R_vecs_j, 2, dim=1, newdim=2)
                    angular_aev.append(self.angular_subaev(R_vecs_pairs))
                except AEVIsZero:
                    # If unable to find pair of neighbor atoms with desired species, fill the subAEV with zeros
                    angular_aev.append(zero_subaev)
            angular_aev = torch.cat(angular_aev, dim=1)
            angular_aevs.append(angular_aev)
        angular_aevs = torch.stack(angular_aevs, dim=1)

        return angular_aevs

    def forward(self, coordinates, species):
        # For the docstring of this method, refer to the base class
        radial_aevs, neighbors = self.compute_neighborlist(coordinates, species)
        return radial_aevs, self.compute_aev_using_neighborlist(coordinates, species, neighbors)

    def export_radial_subaev_onnx(self, filename):
        """Export the operation that compute radial subaev into onnx format

        Parameters
        ----------
        filename : string
            Name of the file to store exported networks.
        """
        class M(torch.nn.Module):
            def __init__(self, outerself):
                super(M, self).__init__()
                self.outerself = outerself

            def forward(self, center, neighbors):
                return self.outerself.radial_subaev(center, neighbors)
        dummy_center = torch.randn(1, 3)
        dummy_neighbors = torch.randn(1, 5, 3)
        torch.onnx.export(M(self), (dummy_center, dummy_neighbors), filename)