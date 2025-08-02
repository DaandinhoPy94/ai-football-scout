# blockchain/nft_minter.py
from web3 import Web3
from eth_account import Account
import ipfshttpclient
import json
from typing import Dict, List
import asyncio

class HighlightNFTMinter:
    """
    Mint NFTs for special football moments
    """
    
    def __init__(
        self,
        web3_provider: str,
        contract_address: str,
        private_key: str,
        ipfs_api: str = "/dns/ipfs.infura.io/tcp/5001/https"
    ):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.contract_address = contract_address
        self.account = Account.from_key(private_key)
        
        # IPFS client
        self.ipfs = ipfshttpclient.connect(ipfs_api)
        
        # Load contract ABI
        self.contract = self._load_contract()
        
    def _load_contract(self):
        """
        Load NFT contract
        """
        abi = [
            {
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "tokenURI", "type": "string"}
                ],
                "name": "mintHighlight",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        return self.w3.eth.contract(
            address=self.contract_address,
            abi=abi
        )
    
    async def mint_highlight(
        self,
        video_clip_path: str,
        metadata: Dict,
        recipient_address: str
    ) -> Dict[str, str]:
        """
        Mint NFT for a highlight
        """
        # Upload video to IPFS
        video_hash = await self._upload_to_ipfs(video_clip_path)
        
        # Create metadata
        nft_metadata = {
            "name": metadata["title"],
            "description": metadata["description"],
            "image": f"ipfs://{video_hash}",
            "attributes": [
                {"trait_type": "Player", "value": metadata["player_name"]},
                {"trait_type": "Team", "value": metadata["team"]},
                {"trait_type": "Match", "value": metadata["match"]},
                {"trait_type": "Event Type", "value": metadata["event_type"]},
                {"trait_type": "Rarity", "value": metadata["rarity"]},
                {"trait_type": "Timestamp", "value": metadata["timestamp"]}
            ],
            "properties": {
                "files": [
                    {
                        "uri": f"ipfs://{video_hash}",
                        "type": "video/mp4"
                    }
                ],
                "category": "video"
            }
        }
        
        # Upload metadata to IPFS
        metadata_hash = await self._upload_json_to_ipfs(nft_metadata)
        
        # Mint NFT
        tx_hash = await self._mint_nft(
            recipient_address,
            f"ipfs://{metadata_hash}"
        )
        
        return {
            "transaction_hash": tx_hash,
            "token_uri": f"ipfs://{metadata_hash}",
            "video_ipfs": f"ipfs://{video_hash}",
            "opensea_url": self._get_opensea_url(tx_hash)
        }
    
    async def _upload_to_ipfs(self, file_path: str) -> str:
        """
        Upload file to IPFS
        """
        result = self.ipfs.add(file_path)
        return result['Hash']
    
    async def _mint_nft(
        self,
        to_address: str,
        token_uri: str
    ) -> str:
        """
        Execute NFT minting transaction
        """
        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        tx = self.contract.functions.mintHighlight(
            to_address,
            token_uri
        ).build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return receipt['transactionHash'].hex()

# Smart contract
"""
// contracts/FootballHighlights.sol
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract FootballHighlights is ERC721, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    mapping(uint256 => HighlightData) public highlights;
    
    struct HighlightData {
        string playerName;
        string matchId;
        uint256 timestamp;
        string eventType;
        uint8 rarityScore;
    }
    
    event HighlightMinted(
        uint256 indexed tokenId,
        address indexed owner,
        string playerName,
        string eventType
    );
    
    constructor() ERC721("Football Highlights", "GOAL") {}
    
    function mintHighlight(
        address to,
        string memory tokenURI,
        HighlightData memory data
    ) public onlyOwner returns (uint256) {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(to, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        highlights[newTokenId] = data;
        
        emit HighlightMinted(
            newTokenId,
            to,
            data.playerName,
            data.eventType
        );
        
        return newTokenId;
    }
}
"""